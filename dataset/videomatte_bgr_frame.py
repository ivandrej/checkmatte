import os
import random

from PIL import Image

from dataset.precaptured_bgr_augmentation import PrecapturedBgrAugmentation
from dataset.videomatte import VideoMatteDataset

"""
    For each frame in the composited video, assigns a frame from the pre-captured background video. 
"""
class VideoMattePrecapturedBgrDataset(VideoMatteDataset):
    def __getitem__(self, idx):
        bgrs, precaptured_bgrs = self._get_random_video_background()

        fgrs, phas = self._get_videomatte(idx)

        return self.transform(fgrs, phas, bgrs, precaptured_bgrs)

    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        precaptured_bgrs = []
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i

            # TODO: Extract in common method
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                bgr = self._downsample_if_needed(bgr.convert('RGB'))
            bgrs.append(bgr)

            # get unaligned background frame
            offset = self._get_random_background_frame_offset()
            precaptured_frame = self.background_video_frames[clip_idx][(frame_idx_t + offset) % frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, precaptured_frame)) as precaptured_bgr:
                precaptured_bgr = self._downsample_if_needed(precaptured_bgr.convert('RGB'))
            precaptured_bgrs.append(precaptured_bgr)

        return bgrs, precaptured_bgrs

    """
        Returns an offset in the interval [-max_offset, max_offset]
    """
    def _get_random_background_frame_offset(self):
        max_offset = 50
        return random.randint(-max_offset, max_offset)


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
