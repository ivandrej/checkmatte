import os
import random

from PIL import Image
import numpy as np
from dataset.precaptured_bgr_augmentation import PrecapturedBgrAugmentation
from dataset.videomatte import VideoMatteDataset

"""
    For each frame in the composited video, assigns a frame from the pre-captured background video. 
"""
class VideoMattePrecapturedBgrDataset(VideoMatteDataset):
    def __init__(self, offset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset

    def __getitem__(self, idx):
        bgrs, precaptured_bgrs = self._get_random_video_background()

        fgrs, phas = self._get_videomatte(idx)

        return self.transform(fgrs, phas, bgrs, precaptured_bgrs)

    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        # clip = self.background_video_clips[clip_idx]
        bgrs = []
        precaptured_bgrs = []

        offset_generator = TemporalOffset(self.offset)
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i

            frame_idx_t = frame_idx_t % frame_count
            bgr = self.read_frame(clip_idx, frame_idx_t)
            bgrs.append(bgr)

            # get unaligned background frame
            offset = offset_generator.get_frame_offset()
            # if offset frame reached end of bgr video, just return last frame
            bgr_frame_idx_t = min(frame_count - 1, frame_idx_t + offset)
            precaptured_bgr = self.read_frame(clip_idx, frame_idx_t)
            precaptured_bgrs.append(precaptured_bgr)

        return bgrs, precaptured_bgrs

    def read_frame(self, clip_idx, frame_idx_t):
        frame = self.background_video_frames[clip_idx][frame_idx_t]
        clip = self.background_video_clips[clip_idx]
        with Image.open(os.path.join(self.background_video_dir, clip, frame)) as frm:
            frm = self._downsample_if_needed(frm.convert('RGB'))
        return frm

"""
    Fixed offset given by max_offset (bgr is always faster).
"""
class TemporalOffset:
    def __init__(self, max_offset):
        self.max_offset = max_offset

    def get_frame_offset(self):
        return self.max_offset

class VideoMattePrecapturedBgrTrainAugmentation(PrecapturedBgrAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.3,
            prob_bgr_affine=0.3,
            prob_noise=0.1,
            prob_color_jitter=0.3,
            prob_grayscale=0.02,
            prob_sharpness=0.1,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
            random_sized_crop=False
        )


class VideoMattePrecapturedBgrValidAugmentation(PrecapturedBgrAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0,
            prob_bgr_affine=0,
            prob_noise=0,
            prob_color_jitter=0,
            prob_grayscale=0,
            prob_sharpness=0,
            prob_blur=0,
            prob_hflip=0,
            prob_pause=0,
            random_sized_crop=False,
            static_affine=False
        )
