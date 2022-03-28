"""
"""

import argparse
import os
import random

import numpy as np
import torch
from PIL import Image
from torch import distributed as dist
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import center_crop
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.augmentation import ValidFrameSampler, TrainFrameSampler
from dataset.videomatte_bgr_frame import VideoMattePrecapturedBgrDataset, VideoMattePrecapturedBgrTrainAugmentation, \
    VideoMattePrecapturedBgrValidAugmentation
from evaluation.evaluation_metrics import MetricMAD
from train_config import BGR_FRAME_DATA_PATHS
from train_loss import pha_loss
from utils import tensor_memory_usage
from visualization.visualize_attention import TrainVisualizer, calc_avg_dha


class AbstractAttentionTrainer:
    def __init__(self, rank, world_size):
        torch.autograd.set_detect_anomaly(True)
        self.parse_args()
        self.init_distributed(rank, world_size)
        self.init_datasets()
        self.init_model()
        self.init_writer()
        self.train()
        self.cleanup()

    def base_args(self):
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
        parser.add_argument('--transformations', type=str, choices=['none', 'person_only', 'same_person_bgr'],
                            default='none')
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
        return parser

    def parse_args(self):
        parser = self.base_args()
        self.custom_args(parser)
        self.args = parser.parse_args()

    def custom_args(self, parser):
        return parser

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
        self.dataset_valid = VideoMattePrecapturedBgrDataset(
            videomatte_dir=BGR_FRAME_DATA_PATHS['videomatte']['valid'],
            background_video_dir=BGR_FRAME_DATA_PATHS['DVM']['valid'],
            size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
            seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
            seq_sampler=ValidFrameSampler(),
            transform=VideoMattePrecapturedBgrValidAugmentation(
                size_hr if self.args.train_hr else self.args.resolution_lr),
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

        self.random_bgr_path = BGR_FRAME_DATA_PATHS["room"]

    def init_network(self):
        raise Exception("This is a base class, pls extend")

    def init_model(self):
        self.log('Initializing model')
        self.init_network()

        # for param in self.model.backbone.parameters():
        #     param.requires_grad = False

        if self.args.checkpoint:
            self.log(f'Restoring from checkpoint: {self.args.checkpoint}')
            self.log(self.model.load_state_dict(
                torch.load(self.args.checkpoint, map_location=f'cuda:{self.rank}')))

        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model_ddp = DDP(self.model, device_ids=[self.rank], broadcast_buffers=False, find_unused_parameters=True)

        self.optimizer = Adam(self.param_lrs)
        self.scaler = GradScaler()

    def init_writer(self):
        if self.rank == 0:
            self.log('Initializing writer')
            self.writer = SummaryWriter(self.args.log_dir)

            self.attention_visualizer = TrainVisualizer(self.writer)

    def train(self):
        for epoch in range(self.args.epoch_start, self.args.epoch_end):
            self.epoch = epoch
            self.step = epoch * len(self.dataloader_lr_train)
            if not self.args.disable_validation:
                # self.validate()
                self.validate_hard()

            self.log(f'Training epoch: {epoch}')
            print("Step at start of this epoch: ", self.step)
            print("Training samples: ", len(self.dataloader_lr_train))
            for true_fgr, true_pha, true_bgr, precaptured_bgr in \
                    tqdm(self.dataloader_lr_train, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                self.train_mat(true_fgr, true_pha, true_bgr, precaptured_bgr, downsample_ratio=1, tag='lr')

                if self.step % self.args.checkpoint_save_interval == 0:
                    self.save()

                self.step += 1
                # print("Step: ", self.step)

    def train_mat(self, true_fgr, true_pha, true_bgr, precaptured_bgr, downsample_ratio, tag):
        true_fgr = true_fgr.to(self.rank, non_blocking=True)
        true_pha = true_pha.to(self.rank, non_blocking=True)
        true_bgr = true_bgr.to(self.rank, non_blocking=True)
        # Uncomment for random cropping of composited images
        # true_fgr, true_pha, true_bgr = self.random_crop(true_fgr, true_pha, true_bgr)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)

        if self.step == 0:
            print("Training batch shape: ", true_src.shape)
            print("True fgr memory usage: ", tensor_memory_usage(true_fgr))
            print("True pha memory usage: ", tensor_memory_usage(true_pha))
            print("True bgr memory usage: ", tensor_memory_usage(true_bgr))
            print("True src memory usage: ", tensor_memory_usage(true_src))

        with autocast(enabled=not self.args.disable_mixed_precision):
            _, pred_pha, attention_intermediate = self.model_ddp(true_src, precaptured_bgr,
                                                                 downsample_ratio=downsample_ratio,
                                                                 return_intermediate=True)[:3]
            attention = attention_intermediate['attention']
            loss = pha_loss(pred_pha, true_pha)

        self.scaler.scale(loss['total']).backward()

        self.log_grad_norms()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.rank == 0 and self.step % self.args.log_train_loss_interval == 0:
            for loss_name, loss_value in loss.items():
                self.writer.add_scalar(f'train_{tag}_{loss_name}', loss_value, self.step)

            train_mad = MetricMAD()(pred_pha, true_pha)
            self.writer.add_scalar(f'train_{tag}_pha_mad', train_mad, self.step)

        if self.rank == 0 and self.step % self.args.log_train_images_interval == 0:
            self.log_train_predictions(precaptured_bgr, pred_pha, true_pha, true_src)
            self.attention_visualizer(attention[0], true_src[0].detach().cpu(), precaptured_bgr[0].detach().cpu(),
                                      self.step, 'train')
            # Plot min and max for energy to ensure no overflow or underflow
            self.writer.add_scalar(f'train_{tag}_energy_min', np.min(attention_intermediate['energy']), self.step)
            self.writer.add_scalar(f'train_{tag}_energy_max', np.max(attention_intermediate['energy']), self.step)

            train_avg_dha = calc_avg_dha(attention)
            self.writer.add_scalar(f'train_{tag}_attention_dha', train_avg_dha, self.step)
            self.test_on_random_bgr(true_src, true_pha, pred_pha, downsample_ratio=1, tag='train')

    def test_on_random_bgr(self, true_src, true_pha, pred_pha, downsample_ratio, tag):
        self.model_ddp.eval()
        with torch.no_grad():
            random_bgr = self.read_random_bgr(true_src.shape).unsqueeze(0)
            random_bgr = random_bgr.repeat(true_src.shape[0], 1, 1, 1, 1)
            random_bgr = random_bgr.to(self.rank, non_blocking=True)

            _, randombgr_pred_pha, attention_intermediate = self.model_ddp(true_src,
                                                              random_bgr,
                                                              downsample_ratio=downsample_ratio,
                                                              return_intermediate=True)[:3]
            attention = attention_intermediate['attention']

            self.log_randombgr_metrics(attention, pred_pha, randombgr_pred_pha, tag, true_pha,
                                       true_src, random_bgr)

        self.model_ddp.train()

    def validate_hard(self):
        if self.rank == 0:
            self.log(f'Validating hard at the start of epoch: {self.epoch}')
            self.model_ddp.eval()
            total_loss, total_count = 0, 0
            randombgr_and_correctbgr_total_mad, total_mad, randombgr_total_mad = 0, 0, 0
            total_dha, randombgr_total_dha = 0, 0
            attentions_total_mad = 0
            pred_phas = []
            true_srcs = []
            precaptured_bgrs = []
            randombgr_pred_phas = []
            randombgrs = []
            attention_to_log = None
            randombgr_attention_to_log = None
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

                        # Only read random bgr once for performance
                        if i == 0:
                            random_bgr = self.read_random_bgr(true_src.shape).to(self.rank,
                                                                                 non_blocking=True).unsqueeze(0)
                            random_bgr = random_bgr.repeat(true_src.shape[0], 1, 1, 1, 1)

                        # The last batch does not have exactly (batch_size, seq_len) dimensions
                        random_bgr = random_bgr[:true_src.shape[0], :true_src.shape[1], :, :, :]

                        # random_noise_bgr = torch.rand_like(true_src).to(self.rank, non_blocking=True)

                        assert (random_bgr.shape == true_src.shape)
                        if total_count == 0:  # only print once
                            print("Validation hard batch shape: ", true_src.shape)

                        batch_size = true_src.size(0)
                        _, pred_pha, attention_intermediate = self.model(true_src, precaptured_bgr,
                                                                         return_intermediate=True)[:3]
                        attention = attention_intermediate['attention']

                        total_loss += pha_loss(pred_pha, true_pha)['total'].item() * batch_size
                        total_mad += MetricMAD()(pred_pha, true_pha) * batch_size
                        total_count += batch_size

                        _, randombgr_pred_pha, attention_intermediate = self.model(true_src,
                                                                                   random_bgr,
                                                                                   return_intermediate=True)[:3]
                        randombgr_attention = attention_intermediate['attention']
                        randombgr_total_mad += MetricMAD()(randombgr_pred_pha, true_pha) * batch_size
                        randombgr_and_correctbgr_total_mad += MetricMAD()(randombgr_pred_pha, pred_pha) * batch_size

                        # _, randomnoisebgr_pred_pha, randomnoisebgr_attention = self.model(true_src, random_noise_bgr)[:3]

                        # Only log attention for the first sequence
                        if i == 0:
                            attention_to_log = attention
                            randombgr_attention_to_log = randombgr_attention
                            # randomnoisebgr_attention_to_log = randomnoisebgr_attention

                        total_dha += calc_avg_dha(attention) * batch_size
                        randombgr_total_dha += calc_avg_dha(randombgr_attention) * batch_size
                        # print("Random bgr dha for batch: ", randombgr_total_dha / batch_size)
                        attentions_total_mad += np.mean(np.absolute(attention - randombgr_attention)) * batch_size

                        if i == 0:  # only show first batch
                            pred_phas.append(pred_pha)
                            true_srcs.append(true_src)
                            precaptured_bgrs.append(precaptured_bgr)
                            randombgr_pred_phas.append(randombgr_pred_pha)
                            randombgrs.append(random_bgr)
                        i += 1
            pred_phas = pred_phas[0]
            true_srcs = true_srcs[0]
            precaptured_bgrs = precaptured_bgrs[0]
            randombgr_pred_phas = randombgr_pred_phas[0]
            randombgrs = randombgrs[0]

            if self.rank == 0:
                self.log_valid_images(attention_to_log, precaptured_bgrs, pred_phas, randombgr_attention_to_log,
                                      randombgr_pred_phas, true_srcs, randombgrs)

            self.log_valid_metrics(attentions_total_mad, randombgr_and_correctbgr_total_mad, randombgr_total_dha,
                                   randombgr_total_mad, total_count, total_dha, total_loss, total_mad)

            self.model_ddp.train()
        dist.barrier()

    # LOGGING HELPERS

    def log_train_predictions(self, precaptured_bgr, pred_pha, true_pha, true_src):
        # self.writer.add_image(f'train_{tag}_pred_fgr', make_grid(pred_fgr.flatten(0, 1), nrow=pred_fgr.size(1)),
        #                       self.step)
        self.writer.add_image(f'train_pred_pha', make_grid(pred_pha.flatten(0, 1), nrow=pred_pha.size(1)),
                              self.step)
        # self.writer.add_image(f'train_{tag}_true_fgr', make_grid(true_fgr.flatten(0, 1), nrow=true_fgr.size(1)),
        #                       self.step)
        self.writer.add_image(f'train_true_pha', make_grid(true_pha.flatten(0, 1), nrow=true_pha.size(1)),
                              self.step)
        self.writer.add_image(f'train_true_src', make_grid(true_src.flatten(0, 1), nrow=true_src.size(1)),
                              self.step)
        self.writer.add_image(f'train_precaptured_bgr', make_grid(precaptured_bgr.flatten(0, 1),
                                                                  nrow=precaptured_bgr.size(1)),
                              self.step)

    def log_valid_metrics(self, attentions_total_mad, randombgr_and_correctbgr_total_mad, randombgr_total_dha,
                          randombgr_total_mad, total_count, total_dha, total_loss, total_mad):
        # Loss
        avg_loss = total_loss / total_count
        self.log(f'Hard validation set average loss: {avg_loss}')
        # MAD
        avg_mad = total_mad / total_count
        avg_randombgr_mad = randombgr_total_mad / total_count
        avg_randombgr_and_correctbgr_mad = randombgr_and_correctbgr_total_mad / total_count
        avg_attentions_mad = attentions_total_mad / total_count
        self.log(f'Hard validation set MAD: {avg_mad}')
        self.writer.add_scalar('hard_valid_mad', avg_mad, self.step)
        self.writer.add_scalar('hard_valid_randombgr_and_correctbgr_mad', avg_randombgr_and_correctbgr_mad,
                               self.step)
        self.writer.add_scalar('hard_valid_randombgr_mad', avg_randombgr_mad, self.step)
        self.writer.add_scalar('hard_valid_attentions_mad', avg_attentions_mad, self.step)
        # DHA
        avg_dha = total_dha / total_count
        randombgr_avg_dha = randombgr_total_dha / total_count
        self.writer.add_scalar('hard_valid_attention_dha', avg_dha, self.step)
        self.writer.add_scalar('hard_valid_randombgr_attention_dha', randombgr_avg_dha, self.step)

    def log_valid_images(self, attention_to_log, precaptured_bgrs, pred_phas, randombgr_attention_to_log,
                         randombgr_pred_phas, true_srcs, randombgrs):
        self.writer.add_image(f'hard_valid_pred_pha',
                              make_grid(pred_phas.flatten(0, 1), nrow=pred_phas.size(1)),
                              self.step)
        self.writer.add_image(f'hard_valid_true_src',
                              make_grid(true_srcs.flatten(0, 1), nrow=true_srcs.size(1)),
                              self.step)
        self.writer.add_image(f'hard_valid_precaptured_bgr',
                              make_grid(precaptured_bgrs.flatten(0, 1), nrow=precaptured_bgrs.size(1)),
                              self.step)
        # self.writer.add_image(f'hard_valid_randomnoisebgr',
        #                       make_grid(random_noise_bgr.flatten(0, 1), nrow=random_noise_bgr.size(1)),
        #                       self.step)
        self.writer.add_image(f'hard_valid_pred_pha_wrongbgr',
                              make_grid(randombgr_pred_phas.flatten(0, 1), nrow=randombgr_pred_phas.size(1)),
                              self.step)
        self.attention_visualizer(attention_to_log[0],
                                  true_srcs[0].detach().cpu(), precaptured_bgrs[0].detach().cpu(),
                                  self.step, 'hard_valid')
        self.attention_visualizer(randombgr_attention_to_log[0],
                                  true_srcs[0].detach().cpu(), randombgrs[0].detach().cpu(),
                                  self.step, 'hard_valid_randombgr')
        # self.attention_visualizer(randomnoisebgr_attention_to_log, self.step, 'hard_valid_randomnoisebgr')

    def log_grad_norms(self):
        bgr_encoder_grad_norm = torch.linalg.vector_norm(
            torch.flatten(self.model_ddp.module.bgr_backbone_grads()))
        self.writer.add_scalar(f'bgr_encoder_grad_norm', bgr_encoder_grad_norm, self.step)
        person_encoder_grad_norm = torch.linalg.vector_norm(
            torch.flatten(self.model_ddp.module.backbone_grads()))
        self.writer.add_scalar(f'person_encoder_grad_norm', person_encoder_grad_norm, self.step)
        attention_key_grad_norm = torch.linalg.vector_norm(
            torch.flatten(self.model_ddp.module.attention_module().key_conv.weight.grad))
        self.writer.add_scalar(f'attention_key_grad_norm', attention_key_grad_norm, self.step)
        attention_query_grad_norm = torch.linalg.vector_norm(
            torch.flatten(self.model_ddp.module.attention_module().query_conv.weight.grad))
        self.writer.add_scalar(f'attention_query_grad_norm', attention_query_grad_norm, self.step)

    """
        Logs MAD metric
        Logs image predictions and attention maps 
    """
    def log_randombgr_metrics(self, attention, pred_pha, randombgr_pred_pha, tag, true_pha, true_src, randombgr):
        random_bgr_mad = MetricMAD()(randombgr_pred_pha, true_pha)
        self.writer.add_scalar(f'{tag}_wrongbgr_mad', random_bgr_mad, self.step)
        mad_random_and_correct_pha = MetricMAD()(randombgr_pred_pha, pred_pha)
        self.writer.add_scalar(f'{tag}_wrongbgr_and_correctbgr_mad', mad_random_and_correct_pha, self.step)
        self.writer.add_image(f'{tag}_pred_pha_wrongbgr',
                              make_grid(randombgr_pred_pha.flatten(0, 1), nrow=randombgr_pred_pha.size(1)),
                              self.step)
        self.attention_visualizer(attention[0], true_src[0].detach().cpu(), randombgr[0].detach().cpu(), self.step, f'{tag}_wrongbgr')
        train_wrongbgr_avg_dha = calc_avg_dha(attention)
        self.writer.add_scalar(f'{tag}_wrongbgr_attention_dha', train_wrongbgr_avg_dha, self.step)

    # MISCELLANEOUS UTILS

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

    def read_random_bgr(self, true_shape):
        _, T, _, H, W = true_shape
        frames = []
        i = 0
        for frameid in sorted(os.listdir(self.random_bgr_path)):
            if i == self.args.seq_length_lr:
                break

            with Image.open(os.path.join(self.random_bgr_path, frameid)) as frm:
                frames.append(frm.convert('RGB').resize((W, H)))
            i += 1

        frames = torch.stack([transforms.functional.to_tensor(frm) for frm in frames])
        return frames

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