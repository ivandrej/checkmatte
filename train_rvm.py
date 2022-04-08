"""
# First update `train_config.py` to set paths to your dataset locations.

# You may want to change `--num-workers` according to your machine's memory.
# The default num-workers=8 may cause dataloader to exit unexpectedly when
# machine is out of memory.

# Stage 1
python train_rvm.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 15 \
    --learning-rate-backbone 0.0001 \
    --learning-rate-aspp 0.0002 \
    --learning-rate-decoder 0.0002 \
    --learning-rate-refiner 0 \
    --checkpoint-dir checkpoint/stage1 \
    --log-dir log/stage1 \
    --epoch-start 0 \
    --epoch-end 20

# Stage 2
python train_rvm.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 50 \
    --learning-rate-backbone 0.00005 \
    --learning-rate-aspp 0.0001 \
    --learning-rate-decoder 0.0001 \
    --learning-rate-refiner 0 \
    --checkpoint checkpoint/stage1/epoch-19.pth \
    --checkpoint-dir checkpoint/stage2 \
    --log-dir log/stage2 \
    --epoch-start 20 \
    --epoch-end 22
    
# Stage 3
python train_rvm.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --train-hr \
    --resolution-lr 512 \
    --resolution-hr 2048 \
    --seq-length-lr 40 \
    --seq-length-hr 6 \
    --learning-rate-backbone 0.00001 \
    --learning-rate-aspp 0.00001 \
    --learning-rate-decoder 0.00001 \
    --learning-rate-refiner 0.0002 \
    --checkpoint checkpoint/stage2/epoch-21.pth \
    --checkpoint-dir checkpoint/stage3 \
    --log-dir log/stage3 \
    --epoch-start 22 \
    --epoch-end 23

# Stage 4
python train_rvm.py \
    --model-variant mobilenetv3 \
    --dataset imagematte \
    --train-hr \
    --resolution-lr 512 \
    --resolution-hr 2048 \
    --seq-length-lr 40 \
    --seq-length-hr 6 \
    --learning-rate-backbone 0.00001 \
    --learning-rate-aspp 0.00001 \
    --learning-rate-decoder 0.00005 \
    --learning-rate-refiner 0.0002 \
    --checkpoint checkpoint/stage3/epoch-22.pth \
    --checkpoint-dir checkpoint/stage4 \
    --log-dir log/stage4 \
    --epoch-start 23 \
    --epoch-end 28
"""


import argparse
import os
import random

import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import center_crop
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.augmentation import ValidFrameSampler, TrainFrameSampler
from dataset.videomatte import (
    VideoMatteDataset,
    VideoMatteSpecializedNoAugmentation,
)
from evaluation.evaluation_metrics import MetricMAD
from model.rvm import MattingNetwork
from train_config import BGR_FRAME_DATA_PATHS
from train_loss import matting_loss, pha_loss
from utils import tensor_memory_usage, model_size


class Trainer:
    def __init__(self, rank, world_size):
        self.parse_args()
        self.init_distributed(rank, world_size)
        self.init_datasets()
        self.init_model()
        self.init_writer()
        self.train()
        self.cleanup()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Model
        parser.add_argument('--model-variant', type=str, required=True, choices=['mobilenetv3', 'mobilenetv3reduced', 'resnet50'])
        # Matting dataset
        parser.add_argument('--dataset', type=str, required=True, choices=['videomatte', 'imagematte'])
        parser.add_argument('--videomatte-clips', type=int, default=-1)
        # Learning rate
        parser.add_argument('--learning-rate-backbone', type=float, required=True)
        parser.add_argument('--learning-rate-aspp', type=float, required=True)
        parser.add_argument('--learning-rate-decoder', type=float, required=True)
        parser.add_argument('--learning-rate-refiner', type=float, required=True)
        # Training setting
        parser.add_argument('--train-hr', action='store_true')
        parser.add_argument('--resolution-lr', type=int, default=512)
        parser.add_argument('--resolution-hr', type=int, default=2048)
        parser.add_argument('--seq-length-lr', type=int, required=True)
        parser.add_argument('--seq-length-hr', type=int, default=6)
        parser.add_argument('--downsample-ratio', type=float, default=0.25)
        parser.add_argument('--batch-size-per-gpu', type=int, default=1)
        parser.add_argument('--num-workers', type=int, default=8)
        parser.add_argument('--epoch-start', type=int, default=0)
        parser.add_argument('--epoch-end', type=int, default=16)
        # Tensorboard logging
        parser.add_argument('--log-dir', type=str, required=True)
        parser.add_argument('--log-train-loss-interval', type=int, default=20)
        parser.add_argument('--log-train-images-interval', type=int, default=500)
        # Checkpoint loading and saving
        parser.add_argument('--checkpoint', type=str)
        parser.add_argument('--checkpoint-dir', type=str, required=True)
        parser.add_argument('--checkpoint-save-interval', type=int, default=500)
        # Distributed
        parser.add_argument('--distributed-addr', type=str, default='localhost')
        parser.add_argument('--distributed-port', type=str, default='12355')
        # Debugging
        parser.add_argument('--disable-progress-bar', action='store_true')
        parser.add_argument('--disable-validation', action='store_true')
        parser.add_argument('--disable-mixed-precision', action='store_true')
        self.args = parser.parse_args()
        
    def init_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.log('Initializing distributed')
        os.environ['MASTER_ADDR'] = self.args.distributed_addr
        os.environ['MASTER_PORT'] = self.args.distributed_port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    def init_datasets(self):
        self.log('Initializing matting datasets')
        size_hr = (self.args.resolution_hr, self.args.resolution_hr)
        size_lr = (self.args.resolution_lr, self.args.resolution_lr)
        
        # Matting datasets:
        self.dataset_lr_train = VideoMatteDataset(
            videomatte_dir=BGR_FRAME_DATA_PATHS['videomatte']['train'],
            background_video_dir=BGR_FRAME_DATA_PATHS['DVM']['train'],
            size=self.args.resolution_lr,
            seq_length=self.args.seq_length_lr,
            seq_sampler=TrainFrameSampler(),
            transform=VideoMatteSpecializedNoAugmentation(self.args.resolution_lr),
            max_videomatte_clips=self.args.videomatte_clips)
        # if self.args.train_hr:
        #     self.dataset_hr_train = VideoMatteDataset(
        #         videomatte_dir=RVM_DATA_PATHS['videomatte']['train'],
        #         background_video_dir=RVM_DATA_PATHS['background_videos']['train'],
        #         size=self.args.resolution_hr,
        #         seq_length=self.args.seq_length_hr,
        #         seq_sampler=TrainFrameSampler(),
        #         transform=VideoMatteValidAugmentation(size_hr))
        self.dataset_valid = VideoMatteDataset(
            videomatte_dir=BGR_FRAME_DATA_PATHS['videomatte']['valid'],
            background_video_dir=BGR_FRAME_DATA_PATHS['DVM']['valid'],
            size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
            seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
            seq_sampler=ValidFrameSampler(),
            transform=VideoMatteSpecializedNoAugmentation(self.args.resolution_lr),
            max_videomatte_clips=-1)
        # Dataset of dynamic backgrounds - harder cases than most of the training samples
        self.dataset_valid_hard = VideoMatteDataset(
            videomatte_dir=BGR_FRAME_DATA_PATHS['videomatte']['valid'],
            background_video_dir=BGR_FRAME_DATA_PATHS['phone_captures']['valid'],
            size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
            seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
            seq_sampler=ValidFrameSampler(),
            transform=VideoMatteSpecializedNoAugmentation(self.args.resolution_lr),
            max_videomatte_clips=-1)

        # Matting dataloaders:
        self.datasampler_lr_train = DistributedSampler(
            dataset=self.dataset_lr_train,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)
        self.dataloader_lr_train = DataLoader(
            dataset=self.dataset_lr_train,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            sampler=self.datasampler_lr_train,
            pin_memory=True)
        if self.args.train_hr:
            self.datasampler_hr_train = DistributedSampler(
                dataset=self.dataset_hr_train,
                rank=self.rank,
                num_replicas=self.world_size,
                shuffle=True)
            self.dataloader_hr_train = DataLoader(
                dataset=self.dataset_hr_train,
                batch_size=self.args.batch_size_per_gpu,
                num_workers=self.args.num_workers,
                sampler=self.datasampler_hr_train,
                pin_memory=True)
        self.dataloader_valid = DataLoader(
            dataset=self.dataset_valid,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            pin_memory=True)
        self.dataloader_valid_hard = DataLoader(
            dataset=self.dataset_valid_hard,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            pin_memory=True)
        
    def init_model(self):
        self.log('Initializing model')
        self.model = MattingNetwork(self.args.model_variant, pretrained_backbone=True).to(self.rank)
        
        if self.args.checkpoint:
            self.log(f'Restoring from checkpoint: {self.args.checkpoint}')
            self.log(self.model.load_state_dict(
                torch.load(self.args.checkpoint, map_location=f'cuda:{self.rank}')))
            
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model_ddp = DDP(self.model, device_ids=[self.rank], broadcast_buffers=False, find_unused_parameters=True)
        self.optimizer = Adam([
            {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
            {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp},
            {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
            {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
        ])
        self.scaler = GradScaler()

        print(f"Model size: {model_size(self.model):.2f} MB")
        
    def init_writer(self):
        if self.rank == 0:
            self.log('Initializing writer')
            self.writer = SummaryWriter(self.args.log_dir)
        
    def train(self):
        for epoch in range(self.args.epoch_start, self.args.epoch_end):
            self.epoch = epoch
            self.step = epoch * len(self.dataloader_lr_train)
            if not self.args.disable_validation:
                self.validate()
                self.validate_hard()
            
            self.log(f'Training epoch: {epoch}')
            for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_lr_train, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                # Low resolution pass
                self.train_mat(true_fgr, true_pha, true_bgr, downsample_ratio=1, tag='lr')

                # High resolution pass
                if self.args.train_hr:
                    true_fgr, true_pha, true_bgr = self.load_next_mat_hr_sample()
                    self.train_mat(true_fgr, true_pha, true_bgr, downsample_ratio=self.args.downsample_ratio, tag='hr')
                    
                if self.step % self.args.checkpoint_save_interval == 0:
                    self.save()
                    
                self.step += 1
                
    def train_mat(self, true_fgr, true_pha, true_bgr, downsample_ratio, tag):
        true_fgr = true_fgr.to(self.rank, non_blocking=True)
        true_pha = true_pha.to(self.rank, non_blocking=True)
        true_bgr = true_bgr.to(self.rank, non_blocking=True)
        # true_fgr, true_pha, true_bgr = self.random_crop(true_fgr, true_pha, true_bgr)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
        
        with autocast(enabled=not self.args.disable_mixed_precision):
            pred_fgr, pred_pha = self.model_ddp(true_src, downsample_ratio=downsample_ratio)[:2]
            loss = pha_loss(pred_pha, true_pha)

        if self.step == 0:
            print("True fgr memory usage: ", tensor_memory_usage(true_fgr))
            print("True pha memory usage: ", tensor_memory_usage(true_pha))
            print("True bgr memory usage: ", tensor_memory_usage(true_bgr))
            print("True src memory usage: ", tensor_memory_usage(true_src))

        self.scaler.scale(loss['total']).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        if self.rank == 0 and self.step % self.args.log_train_loss_interval == 0:
            for loss_name, loss_value in loss.items():
                self.writer.add_scalar(f'train_{tag}_{loss_name}', loss_value, self.step)

            train_mad = MetricMAD()(pred_pha, true_pha)
            self.writer.add_scalar(f'train_{tag}_pha_mad', train_mad, self.step)
            
        if self.rank == 0 and self.step % self.args.log_train_images_interval == 0:
            self.writer.add_image(f'train_{tag}_pred_fgr', make_grid(pred_fgr.flatten(0, 1), nrow=pred_fgr.size(1)), self.step)
            self.writer.add_image(f'train_{tag}_pred_pha', make_grid(pred_pha.flatten(0, 1), nrow=pred_pha.size(1)), self.step)
            self.writer.add_image(f'train_{tag}_true_fgr', make_grid(true_fgr.flatten(0, 1), nrow=true_fgr.size(1)), self.step)
            self.writer.add_image(f'train_{tag}_true_pha', make_grid(true_pha.flatten(0, 1), nrow=true_pha.size(1)), self.step)
            self.writer.add_image(f'train_{tag}_true_src', make_grid(true_src.flatten(0, 1), nrow=true_src.size(1)), self.step)

    def load_next_mat_hr_sample(self):
        try:
            sample = next(self.dataiterator_mat_hr)
        except:
            self.datasampler_hr_train.set_epoch(self.datasampler_hr_train.epoch + 1)
            self.dataiterator_mat_hr = iter(self.dataloader_hr_train)
            sample = next(self.dataiterator_mat_hr)
        return sample

    def validate(self):
        if self.rank == 0:
            self.log(f'Validating at the start of epoch: {self.epoch}')
            self.model_ddp.eval()
            total_loss, total_count = 0, 0
            with torch.no_grad():
                with autocast(enabled=not self.args.disable_mixed_precision):
                    for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_valid, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                        true_fgr = true_fgr.to(self.rank, non_blocking=True)
                        true_pha = true_pha.to(self.rank, non_blocking=True)
                        true_bgr = true_bgr.to(self.rank, non_blocking=True)
                        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
                        batch_size = true_src.size(0)
                        pred_fgr, pred_pha = self.model(true_src)[:2]
                        total_loss += matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)['total'].item() * batch_size
                        total_count += batch_size
            avg_loss = total_loss / total_count
            self.log(f'Validation set average loss: {avg_loss}')
            self.writer.add_scalar('valid_loss', avg_loss, self.step)
            self.model_ddp.train()
        dist.barrier()

    def validate_hard(self):
        if self.rank == 0:
            self.log(f'Validating hard at the start of epoch: {self.epoch}')
            self.model_ddp.eval()
            total_loss, total_count = 0, 0
            total_mad = 0
            pred_phas = []
            true_srcs = []
            i = 0
            with torch.no_grad():
                with autocast(enabled=not self.args.disable_mixed_precision):
                    for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_valid_hard,
                                                                              disable=self.args.disable_progress_bar,
                                                                              dynamic_ncols=True):
                        true_fgr = true_fgr.to(self.rank, non_blocking=True)
                        true_pha = true_pha.to(self.rank, non_blocking=True)
                        true_bgr = true_bgr.to(self.rank, non_blocking=True)
                        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
                        if total_count == 0:  # only print once
                            print("Validation hard batch shape: ", true_src.shape)

                        batch_size = true_src.size(0)
                        pred_fgr, pred_pha = self.model(true_src)[:2]
                        total_loss += matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)[
                                          'total'].item() * batch_size
                        total_mad += MetricMAD()(pred_pha, true_pha) * batch_size
                        total_count += batch_size

                        if i % 12 == 0:  # reduces number of samples to show
                            pred_phas.append(pred_pha)
                            true_srcs.append(true_src)
                        i += 1
            pred_phas = pred_phas[0]
            true_srcs = true_srcs[0]

            if self.rank == 0:
                self.writer.add_image(f'hard_valid_pred_pha',
                                      make_grid(pred_phas.flatten(0, 1), nrow=pred_phas.size(1)),
                                      self.step)
                self.writer.add_image(f'hard_valid_true_src',
                                      make_grid(true_srcs.flatten(0, 1), nrow=true_srcs.size(1)),
                                      self.step)
            avg_loss = total_loss / total_count
            avg_mad = total_mad / total_count

            self.log(f'Hard validation set average loss: {avg_loss}')
            self.log(f'Hard validation set MAD: {avg_mad}')
            self.writer.add_scalar('hard_valid_mad', avg_mad, self.step)
            self.model_ddp.train()
        dist.barrier()
    
    def random_crop(self, *imgs):
        h, w = imgs[0].shape[-2:]
        w = random.choice(range(w // 2, w))
        h = random.choice(range(w // 2, h))
        results = []
        for img in imgs:
            B, T = img.shape[:2]
            img = img.flatten(0, 1)
            img = F.interpolate(img, (max(h, w), max(h, w)), mode='bilinear', align_corners=False)
            img = center_crop(img, (h, w))
            img = img.reshape(B, T, *img.shape[1:])
            results.append(img)
        return results
    
    def save(self):
        if self.rank == 0:
            os.makedirs(self.args.checkpoint_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.args.checkpoint_dir, f'epoch-{self.epoch}.pth'))
            self.log('Model saved')
        dist.barrier()
        
    def cleanup(self):
        dist.destroy_process_group()
        
    def log(self, msg):
        print(f'[GPU{self.rank}] {msg}')
            
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(
        Trainer,
        nprocs=world_size,
        args=(world_size,),
        join=True)
