"""
    Trains the model where we concatenate the background and the person features instead of performing the attention.
    The model is described in model/model_concat_bgr
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
from dataset.videomatte_with_precaptured_bgr import VideoMattePrecapturedBgrDataset, \
    VideoMattePrecapturedBgrTrainAugmentation, \
    VideoMattePrecapturedBgrValidAugmentation
from evaluation.evaluation_metrics import MetricMAD
from model.model_concat_bgr import MattingNetwork
from train_config import BGR_FRAME_DATA_PATHS
from train_loss import matting_loss, segmentation_loss


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
        parser.add_argument('--model-variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
        # Matting dataset
        parser.add_argument('--dataset', type=str, required=True, choices=['videomatte', 'imagematte'])
        parser.add_argument('--videomatte-clips', type=int, required=True)
        # Learning rate
        parser.add_argument('--learning-rate-backbone', type=float, required=True)
        parser.add_argument('--learning-rate-aspp', type=float, required=True)
        parser.add_argument('--learning-rate-decoder', type=float, required=True)
        parser.add_argument('--learning-rate-refiner', type=float, required=True)
        # Training setting
        parser.add_argument('--train-hr', action='store_true')
        parser.add_argument('--pretrained_on_rvm', action='store_true')
        parser.add_argument('--varied-every-n-steps', type=int, default=None)  # no varied bgrs by default
        parser.add_argument('--seg-every-n-steps', type=int, default=None)  # no seg by default
        parser.add_argument('--temporal_offset', type=int, default=0)  # temporal offset between precaptured bgr and src
        # what kind of transformations to apply to bgr and person frames. Default is no transformations
        parser.add_argument('--transformations', type=str, choices=['none', 'person_only', 'same_person_bgr'], default='none')
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
        parser.add_argument('--log-randombgr-mad-interval', type=int, default=None)
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

        if self.args.transformations == 'person_only':
            train_augmentation = VideoMattePrecapturedBgrTrainAugmentation(self.args.resolution_lr)
        elif self.args.transformations == 'same_person_bgr':
            train_augmentation = VideoMattePrecapturedBgrTrainAugmentation(self.args.resolution_lr)
        elif self.args.transformations == 'none':
            train_augmentation = VideoMattePrecapturedBgrValidAugmentation(self.args.resolution_lr)
        else:
            raise Exception(f'Transformation "{self.args.transformations}" not supported')

        # Matting datasets:
        self.dataset_lr_train = VideoMattePrecapturedBgrDataset(
            videomatte_dir=BGR_FRAME_DATA_PATHS['videomatte']['train'],
            background_video_dir=BGR_FRAME_DATA_PATHS['DVM']['train'],
            size=self.args.resolution_lr,
            seq_length=self.args.seq_length_lr,
            seq_sampler=TrainFrameSampler(),
            transform=train_augmentation,
            max_videomatte_clips=self.args.videomatte_clips,
            offset=self.args.temporal_offset)
        # if self.args.train_hr:
        #     self.dataset_hr_train = VideoMattePrecapturedBgrDataset(
        #         videomatte_dir=BGR_FRAME_DATA_PATHS['videomatte']['train'],
        #         background_video_dir=BGR_FRAME_DATA_PATHS['DVM']['train'],
        #         size=self.args.resolution_hr,
        #         seq_length=self.args.seq_length_hr,
        #         seq_sampler=TrainFrameSampler(),
        #         transform=VideoMattePrecapturedBgrTrainAugmentation(size_hr),
        #         max_videomatte_clips=self.args.videomatte_clips
        #     )
        self.dataset_valid = VideoMattePrecapturedBgrDataset(
            videomatte_dir=BGR_FRAME_DATA_PATHS['videomatte']['valid'],
            background_video_dir=BGR_FRAME_DATA_PATHS['DVM']['valid'],
            size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
            seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
            seq_sampler=ValidFrameSampler(),
            transform=VideoMattePrecapturedBgrValidAugmentation(size_hr if self.args.train_hr else self.args.resolution_lr),
            offset=self.args.temporal_offset)

        # Dataset of dynamic backgrounds - harder cases than most of the training samples
        self.dataset_valid_hard = VideoMattePrecapturedBgrDataset(
            videomatte_dir=BGR_FRAME_DATA_PATHS['videomatte']['valid'],
            background_video_dir=BGR_FRAME_DATA_PATHS['phone_captures']['valid'],
            size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
            seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
            seq_sampler=ValidFrameSampler(),
            transform=VideoMattePrecapturedBgrValidAugmentation(
                size_hr if self.args.train_hr else self.args.resolution_lr),
            offset=self.args.temporal_offset)

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
        self.model = MattingNetwork(self.args.model_variant,
                                    pretrained_backbone=True,
                                    pretrained_on_rvm=self.args.pretrained_on_rvm).to(self.rank)

        # for param in self.model.backbone.parameters():
        #     param.requires_grad = False

        if self.args.checkpoint:
            self.log(f'Restoring from checkpoint: {self.args.checkpoint}')
            self.log(self.model.load_state_dict(
                torch.load(self.args.checkpoint, map_location=f'cuda:{self.rank}')))

        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model_ddp = DDP(self.model, device_ids=[self.rank], broadcast_buffers=False, find_unused_parameters=True)
        param_lrs = [{'params': self.model.backbone_bgr.parameters(), 'lr': self.args.learning_rate_backbone},
                     {'params': self.model.aspp_bgr.parameters(), 'lr': self.args.learning_rate_aspp},
                     {'params': self.model.project_concat.parameters(), 'lr': self.args.learning_rate_aspp},
                     {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
                     {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
                     {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
                     {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp}]

        self.optimizer = Adam(param_lrs)
        self.scaler = GradScaler()

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
            print("Step at start of this epoch: ", self.step)
            print("Training samples: ", len(self.dataloader_lr_train))
            for true_fgr, true_pha, true_bgr, precaptured_bgr in \
                    tqdm(self.dataloader_lr_train, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                self.train_mat(true_fgr, true_pha, true_bgr, precaptured_bgr, downsample_ratio=1, tag='lr')

                if self.args.seg_every_n_steps and self.step % self.args.seg_every_n_steps == 0:
                    true_img, true_seg = self.load_next_seg_video_sample()
                    self.train_seg(true_img, true_seg, log_label='seg_video')

                if self.args.log_randombgr_mad_interval and self.step % self.args.log_randombgr_mad_interval == 0:
                    self.test_on_random_bgr(true_fgr, true_pha, true_bgr, downsample_ratio=1)

                if self.step % self.args.checkpoint_save_interval == 0:
                    self.save()

                self.step += 1
                # print("Step: ", self.step)

    def test_on_random_bgr(self, true_fgr, true_pha, true_bgr, downsample_ratio):
        true_fgr = true_fgr.to(self.rank, non_blocking=True)
        true_pha = true_pha.to(self.rank, non_blocking=True)
        true_bgr = true_bgr.to(self.rank, non_blocking=True)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)

        random_bgr = torch.zeros(true_src.shape, device=self.rank)
        _, pred_pha_random_bgr = self.model_ddp(true_src,
                                                random_bgr,
                                                downsample_ratio=downsample_ratio)[:2]
        random_bgr_mad = MetricMAD()(pred_pha_random_bgr, true_pha)
        self.writer.add_scalar(f'random_bgr_mad', random_bgr_mad, self.step)

    def train_mat(self, true_fgr, true_pha, true_bgr, precaptured_bgr, downsample_ratio, tag):
        true_fgr = true_fgr.to(self.rank, non_blocking=True)
        true_pha = true_pha.to(self.rank, non_blocking=True)
        true_bgr = true_bgr.to(self.rank, non_blocking=True)
        # Uncomment for random cropping of composited images
        # true_fgr, true_pha, true_bgr = self.random_crop(true_fgr, true_pha, true_bgr)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)

        if self.step == 0:
            print("Training batch shape: ", true_src.shape)

        with autocast(enabled=not self.args.disable_mixed_precision):
            pred_fgr, pred_pha = self.model_ddp(true_src, precaptured_bgr, downsample_ratio=downsample_ratio)[:2]
            loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)

        self.scaler.scale(loss['total']).backward()

        # TODO: Move to a separate method
        bgr_encoder_grad_norm = torch.linalg.vector_norm(
            torch.flatten(self.model_ddp.module.backbone_bgr.features[16][0].weight.grad))
        self.writer.add_scalar(f'bgr_encoder_grad_norm', bgr_encoder_grad_norm, self.step)

        person_encoder_grad_norm = torch.linalg.vector_norm(
            torch.flatten(self.model_ddp.module.backbone.features[16][0].weight.grad))
        self.writer.add_scalar(f'person_encoder_grad_norm', person_encoder_grad_norm, self.step)

        concat_proj_grad_norm = torch.linalg.vector_norm(
            torch.flatten(self.model_ddp.module.project_concat.conv[0].weight.grad))
        self.writer.add_scalar(f'concat_proj_grad_norm', concat_proj_grad_norm, self.step)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.rank == 0 and self.step % self.args.log_train_loss_interval == 0:
            for loss_name, loss_value in loss.items():
                self.writer.add_scalar(f'train_{tag}_{loss_name}', loss_value, self.step)

        if self.rank == 0 and self.step % self.args.log_train_images_interval == 0:
            # self.writer.add_image(f'train_{tag}_pred_fgr', make_grid(pred_fgr.flatten(0, 1), nrow=pred_fgr.size(1)),
            #                       self.step)
            self.writer.add_image(f'train_{tag}_pred_pha', make_grid(pred_pha.flatten(0, 1), nrow=pred_pha.size(1)),
                                  self.step)
            # self.writer.add_image(f'train_{tag}_true_fgr', make_grid(true_fgr.flatten(0, 1), nrow=true_fgr.size(1)),
            #                       self.step)
            self.writer.add_image(f'train_{tag}_true_pha', make_grid(true_pha.flatten(0, 1), nrow=true_pha.size(1)),
                                  self.step)
            self.writer.add_image(f'train_{tag}_true_src', make_grid(true_src.flatten(0, 1), nrow=true_src.size(1)),
                                  self.step)
            self.writer.add_image(f'train_{tag}_precaptured_bgr', make_grid(precaptured_bgr.flatten(0, 1),
                                                                            nrow=precaptured_bgr.size(1)),
                                  self.step)

    def train_seg(self, true_img, true_seg, log_label):
        true_img = true_img.to(self.rank, non_blocking=True)
        true_seg = true_seg.to(self.rank, non_blocking=True)

        # true_img, true_seg = self.random_crop(true_img, true_seg)
        if self.step == 0:
            print("Segmentation batch size: ", true_seg.shape)

        with autocast(enabled=not self.args.disable_mixed_precision):
            pred_seg = self.model_ddp(true_img, segmentation_pass=True)[0]
            loss = segmentation_loss(pred_seg, true_seg)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.rank == 0 and (self.step - self.step % 2) % self.args.log_train_loss_interval == 0:
            self.writer.add_scalar(f'{log_label}_loss', loss, self.step)

        if self.rank == 0 and (self.step - self.step % 2) % self.args.log_train_images_interval == 0:
            self.writer.add_image(f'{log_label}_pred_seg',
                                  make_grid(pred_seg.flatten(0, 1).float().sigmoid(), nrow=self.args.seq_length_lr),
                                  self.step)
            self.writer.add_image(f'{log_label}_true_seg',
                                  make_grid(true_seg.flatten(0, 1), nrow=self.args.seq_length_lr), self.step)
            self.writer.add_image(f'{log_label}_true_img',
                                  make_grid(true_img.flatten(0, 1), nrow=self.args.seq_length_lr), self.step)

    def load_next_mat_hr_sample(self):
        try:
            sample = next(self.dataiterator_mat_hr)
        except:
            self.datasampler_hr_train.set_epoch(self.datasampler_hr_train.epoch + 1)
            self.dataiterator_mat_hr = iter(self.dataloader_hr_train)
            sample = next(self.dataiterator_mat_hr)
        return sample

    def load_next_varied_bgr_sample(self):
        try:
            sample = next(self.dataiterator_mat_varied)
        except:
            self.datasampler_varied_train.set_epoch(self.datasampler_varied_train.epoch + 1)
            self.dataiterator_mat_varied = iter(self.dataloader_varied_train)
            sample = next(self.dataiterator_mat_varied)
        return sample

    def load_next_seg_video_sample(self):
        try:
            sample = next(self.dataiterator_seg_video)
        except:
            self.datasampler_seg_video.set_epoch(self.datasampler_seg_video.epoch + 1)
            self.dataiterator_seg_video = iter(self.dataloader_seg_video)
            sample = next(self.dataiterator_seg_video)
        return sample

    def validate(self):
        if self.rank == 0:
            self.log(f'Validating at the start of epoch: {self.epoch}')
            self.model_ddp.eval()
            total_loss, total_count = 0, 0
            pred_phas = []
            true_srcs = []
            precaptured_bgrs = []
            i = 0
            with torch.no_grad():
                with autocast(enabled=not self.args.disable_mixed_precision):
                    for true_fgr, true_pha, true_bgr, precaptured_bgr in tqdm(self.dataloader_valid_hard,
                                                                         disable=self.args.disable_progress_bar,
                                                                        dynamic_ncols=True):
                        true_fgr = true_fgr.to(self.rank, non_blocking=True)
                        true_pha = true_pha.to(self.rank, non_blocking=True)
                        true_bgr = true_bgr.to(self.rank, non_blocking=True)
                        precaptured_bgr = precaptured_bgr.to(self.rank, non_blocking=True)
                        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
                        if self.step == 0 and total_count == 0:  # only print once
                            print("Validation batch shape: ", true_src.shape)

                        batch_size = true_src.size(0)
                        pred_fgr, pred_pha = self.model(true_src, precaptured_bgr)[:2]
                        total_loss += matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)['total'].item() * batch_size
                        total_count += batch_size

                        if i % 12 == 0:  # reduces number of samples to show
                            pred_phas.append(pred_pha)
                            true_srcs.append(true_src)
                            precaptured_bgrs.append(precaptured_bgr)
                        i += 1
            mad_error = MetricMAD()(pred_pha, true_pha)
            pred_phas = torch.cat(pred_phas, dim=0)
            true_srcs = torch.cat(true_srcs, dim=0)
            precaptured_bgrs = torch.cat(precaptured_bgrs, dim=0)

            if self.rank == 0:
                self.writer.add_image(f'valid_pred_pha',
                                      make_grid(pred_phas.flatten(0, 1), nrow=pred_phas.size(1)),
                                      self.step)
                self.writer.add_image(f'valid_true_src',
                                      make_grid(true_srcs.flatten(0, 1), nrow=true_srcs.size(1)),
                                      self.step)
                self.writer.add_image(f'valid_precaptured_bgr',
                                      make_grid(precaptured_bgrs.flatten(0, 1), nrow=precaptured_bgrs.size(1)),
                                      self.step)
            avg_loss = total_loss / total_count
            self.log(f'Validation set average loss: {avg_loss}')
            self.log(f'Validation MAD: {mad_error}')
            self.writer.add_scalar('valid_loss', avg_loss, self.step)
            self.writer.add_scalar('valid_mad', mad_error, self.step)
            self.model_ddp.train()
        dist.barrier()

    def validate_hard(self):
        if self.rank == 0:
            self.log(f'Validating hard at the start of epoch: {self.epoch}')
            self.model_ddp.eval()
            total_loss, total_count = 0, 0
            pred_phas = []
            true_srcs = []
            precaptured_bgrs = []
            i = 0
            with torch.no_grad():
                with autocast(enabled=not self.args.disable_mixed_precision):
                    for true_fgr, true_pha, true_bgr, precaptured_bgr in tqdm(self.dataloader_valid_hard,
                                                                              disable=self.args.disable_progress_bar,
                                                                              dynamic_ncols=True):
                        true_fgr = true_fgr.to(self.rank, non_blocking=True)
                        true_pha = true_pha.to(self.rank, non_blocking=True)
                        true_bgr = true_bgr.to(self.rank, non_blocking=True)
                        precaptured_bgr = precaptured_bgr.to(self.rank, non_blocking=True)
                        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
                        if total_count == 0:  # only print once
                            print("Validation hard batch shape: ", true_src.shape)

                        batch_size = true_src.size(0)
                        pred_fgr, pred_pha = self.model(true_src, precaptured_bgr)[:2]
                        total_loss += matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)[
                                          'total'].item() * batch_size
                        total_count += batch_size

                        if i % 12 == 0:  # reduces number of samples to show
                            pred_phas.append(pred_pha)
                            true_srcs.append(true_src)
                            precaptured_bgrs.append(precaptured_bgr)
                        i += 1
            mad_error = MetricMAD()(pred_pha, true_pha)
            pred_phas = pred_phas[0]
            true_srcs = true_srcs[0]
            precaptured_bgrs = precaptured_bgrs[0]

            if self.rank == 0:
                self.writer.add_image(f'hard_valid_pred_pha',
                                      make_grid(pred_phas.flatten(0, 1), nrow=pred_phas.size(1)),
                                      self.step)
                self.writer.add_image(f'hard_valid_true_src',
                                      make_grid(true_srcs.flatten(0, 1), nrow=true_srcs.size(1)),
                                      self.step)
                self.writer.add_image(f'hard_valid_precaptured_bgr',
                                      make_grid(precaptured_bgrs.flatten(0, 1), nrow=precaptured_bgrs.size(1)),
                                      self.step)
            avg_loss = total_loss / total_count
            self.log(f'Hard validation set average loss: {avg_loss}')
            self.log(f'Hard validation set MAD: {mad_error}')
            self.writer.add_scalar('hard_valid_mad', mad_error, self.step)
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
