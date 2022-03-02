import os
import sys

import numpy as np

sys.path.append('..')

from torch import distributed as dist
from torch.cuda.amp import autocast
from torchvision.utils import make_grid
from tqdm import tqdm

from base_attention_trainer import AbstractAttentionTrainer
from evaluation.evaluation_metrics import MetricMAD
from train_loss import pha_loss

from model import model_attention_addition, model_attention_concat

import torch
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

from dataset.augmentation import ValidFrameSampler
from dataset.videomatte_bgr_frame import VideoMattePrecapturedBgrDataset, VideoMattePrecapturedBgrValidAugmentation
from train_config import BGR_FRAME_DATA_PATHS


class ValidationOnlyTrainer(AbstractAttentionTrainer):
    def __init__(self, rank, world_size):
        self.parse_args()
        self.init_distributed(rank, world_size)
        self.init_datasets()
        # self.init_model()
        self.init_writer()
        self.train()
        self.cleanup()

    def custom_args(self, parser):
        parser.add_argument('--model-type', type=str, choices=['addition', 'concat'], default='addition')

    def init_datasets(self):
        self.log('Initializing matting datasets')
        size_hr = (self.args.resolution_hr, self.args.resolution_hr)
        size_lr = (self.args.resolution_lr, self.args.resolution_lr)

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
        if self.args.model_type == 'addition':
            self.model = model_attention_addition.MattingNetwork(self.args.model_variant,
                                                                 pretrained_backbone=True,
                                                                 pretrained_on_rvm=self.args.pretrained_on_rvm).to(
                self.rank)

            self.param_lrs = [
                {'params': self.model.backbone_bgr.parameters(), 'lr': self.args.learning_rate_backbone},
                {'params': self.model.aspp_bgr.parameters(), 'lr': self.args.learning_rate_aspp},
                # {'params': self.model.project_concat.parameters(), 'lr': self.args.learning_rate_aspp},
                {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
                {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
                {'params': self.model.spatial_attention.parameters(),
                 'lr': self.args.learning_rate_backbone},
                {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
                {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp}]
        else:
            self.model = model_attention_concat.MattingNetwork(self.args.model_variant,
                                                               pretrained_backbone=True,
                                                               pretrained_on_rvm=self.args.pretrained_on_rvm).to(
                self.rank)

            self.param_lrs = [
                {'params': self.model.backbone_bgr.parameters(), 'lr': self.args.learning_rate_backbone},
                {'params': self.model.aspp_bgr.parameters(), 'lr': self.args.learning_rate_aspp},
                {'params': self.model.project_concat.parameters(), 'lr': self.args.learning_rate_aspp},
                {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
                {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
                {'params': self.model.spatial_attention.parameters(),
                 'lr': self.args.learning_rate_backbone},
                {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
                {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp}]

    def train(self):
        checkpoint_dir = self.args.checkpoint
        print("Listdir: ", os.listdir(checkpoint_dir))
        for epoch in range(0, len((sorted(os.listdir(checkpoint_dir))))):
            self.epoch = epoch
            self.step = epoch
            self.args.checkpoint = os.path.join(checkpoint_dir, f"epoch-{epoch}.pth")
            self.init_model()
            if not self.args.disable_validation:
                self.validate_hard()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(
        ValidationOnlyTrainer,
        nprocs=world_size,
        args=(world_size,),
        join=True)
