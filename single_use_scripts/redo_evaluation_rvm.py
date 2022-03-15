import os
import sys

sys.path.append('..')
from dataset.videomatte import VideoMatteDataset, VideoMatteSpecializedNoAugmentation
from train_rvm import Trainer


import torch
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

from dataset.augmentation import ValidFrameSampler
from train_config import BGR_FRAME_DATA_PATHS


class ValidationOnlyTrainer(Trainer):
    def __init__(self, rank, world_size):
        self.parse_args()
        self.init_distributed(rank, world_size)
        self.init_datasets()
        # self.init_model()
        self.init_writer()
        self.train()
        self.cleanup()

    def init_datasets(self):
        self.log('Initializing matting datasets')
        size_hr = (self.args.resolution_hr, self.args.resolution_hr)
        size_lr = (self.args.resolution_lr, self.args.resolution_lr)

        # Dataset of dynamic backgrounds - harder cases than most of the training samples
        self.dataset_valid_hard = VideoMatteDataset(
            videomatte_dir=BGR_FRAME_DATA_PATHS['videomatte']['valid'],
            background_video_dir=BGR_FRAME_DATA_PATHS['phone_captures']['valid'],
            size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
            seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
            seq_sampler=ValidFrameSampler(),
            transform=VideoMatteSpecializedNoAugmentation(self.args.resolution_lr),
            max_videomatte_clips=-1)

        self.dataloader_valid_hard = DataLoader(
            dataset=self.dataset_valid_hard,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            pin_memory=True)

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
