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


    def validate_hard(self):
        if self.rank == 0:
            self.log(f'Validating hard at the start of epoch: {self.epoch}')
            self.model_ddp.eval()
            total_loss, total_count = 0, 0
            randombgr_and_correctbgr_total_mad, total_mad, randombgr_total_mad = 0, 0, 0
            attentions_total_mad = 0
            pred_phas = []
            true_srcs = []
            precaptured_bgrs = []
            randombgr_pred_phas = []
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
                            random_bgr = self.read_random_bgr(true_src.shape).to(self.rank, non_blocking=True).unsqueeze(0)
                            random_bgr = random_bgr.repeat(true_src.shape[0], 1, 1, 1, 1)

                        # The last batch does not have exactly (batch_size, seq_len) dimensions
                        random_bgr = random_bgr[:true_src.shape[0], :true_src.shape[1], :, :, :]

                        assert (random_bgr.shape == true_src.shape)
                        if total_count == 0:  # only print once
                            print("Validation hard batch shape: ", true_src.shape)

                        batch_size = true_src.size(0)
                        _, pred_pha, attention = self.model(true_src, precaptured_bgr)[:3]
                        total_loss += pha_loss(pred_pha, true_pha)['total'].item() * batch_size
                        total_mad += MetricMAD()(pred_pha, true_pha) * batch_size
                        total_count += batch_size

                        _, randombgr_pred_pha, randombgr_attention = self.model(true_src, random_bgr)[:3]
                        randombgr_total_mad += MetricMAD()(randombgr_pred_pha, true_pha) * batch_size
                        randombgr_and_correctbgr_total_mad += MetricMAD()(randombgr_pred_pha, pred_pha) * batch_size

                        # Only log attention for the first sequence
                        if i == 0:
                            attention_to_log = attention
                            randombgr_attention_to_log = randombgr_attention

                        attentions_total_mad += np.mean(np.absolute(attention - randombgr_attention)) * batch_size

                        if i == 0:  # only show first batch
                            pred_phas.append(pred_pha)
                            true_srcs.append(true_src)
                            precaptured_bgrs.append(precaptured_bgr)
                            randombgr_pred_phas.append(randombgr_pred_pha)
                        i += 1
            pred_phas = pred_phas[0]
            true_srcs = true_srcs[0]
            precaptured_bgrs = precaptured_bgrs[0]
            randombgr_pred_phas = randombgr_pred_phas[0]

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
                self.writer.add_image(f'hard_valid_pred_pha_wrongbgr',
                                      make_grid(randombgr_pred_phas.flatten(0, 1), nrow=randombgr_pred_phas.size(1)),
                                      self.step)
                self.attention_visualizer(attention_to_log, self.step, 'hard_valid')
                self.attention_visualizer(randombgr_attention_to_log, self.step, 'hard_valid_randombgr')

            avg_loss = total_loss / total_count
            avg_mad = total_mad / total_count
            avg_randombgr_mad = randombgr_total_mad / total_count
            avg_randombgr_and_correctbgr_mad = randombgr_and_correctbgr_total_mad / total_count
            avg_attentions_mad = attentions_total_mad / total_count
            self.log(f'Hard validation set average loss: {avg_loss}')
            self.log(f'Hard validation set MAD: {avg_mad}')
            self.writer.add_scalar('hard_valid_mad', avg_mad, self.step)
            self.writer.add_scalar('hard_valid_randombgr_and_correctbgr_mad', avg_randombgr_and_correctbgr_mad, self.step)
            self.writer.add_scalar('hard_valid_randombgr_mad', avg_randombgr_mad, self.step)
            self.writer.add_scalar('hard_valid_attentions_mad', avg_attentions_mad, self.step)

            self.model_ddp.train()
        dist.barrier()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(
        ValidationOnlyTrainer,
        nprocs=world_size,
        args=(world_size,),
        join=True)
