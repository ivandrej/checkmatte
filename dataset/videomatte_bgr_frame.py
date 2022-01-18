import os
import random

from PIL import Image

from dataset.videomatte import VideoMatteDataset

"""
    For each frame in the composited video, assigns a frame from the pre-captured background video. 
"""
class VideoMattePrecapturedBgrDataset(VideoMatteDataset):
    def __getitem__(self, idx):
        bgrs, precaptured_bgrs = self._get_random_video_background()

        fgrs, phas = self._get_videomatte(idx)

        return self.transform(fgrs, phas, bgrs)

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

    def _get_random_background_frame_offset(self):
        if random.random() < 0.5:
            return -50
        else:
            return 50
