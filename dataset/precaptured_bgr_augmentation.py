import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F

from dataset.augmentation import MotionAugmentation


class PrecapturedBgrAugmentation:
    def __init__(self,
                 size,
                 prob_fgr_affine,
                 prob_bgr_affine,
                 prob_noise,
                 prob_color_jitter,
                 prob_grayscale,
                 prob_sharpness,
                 prob_blur,
                 prob_hflip,
                 prob_pause,
                 static_affine=True,
                 random_sized_crop=True,
                 aspect_ratio_range=(0.9, 1.1)):
        self.size = size
        self.prob_fgr_affine = prob_fgr_affine
        self.prob_bgr_affine = prob_bgr_affine
        self.prob_noise = prob_noise
        self.prob_color_jitter = prob_color_jitter
        self.prob_grayscale = prob_grayscale
        self.prob_sharpness = prob_sharpness
        self.prob_blur = prob_blur
        self.prob_hflip = prob_hflip
        self.prob_pause = prob_pause
        self.static_affine = static_affine
        self.aspect_ratio_range = aspect_ratio_range
        self.random_sized_crop = random_sized_crop

    def __call__(self, fgrs, phas, bgrs, bgrs_):
        # Foreground affine
        if random.random() < self.prob_fgr_affine:
            fgrs, phas = MotionAugmentation.motion_affine(fgrs, phas)

        # Background affine
        if random.random() < self.prob_bgr_affine / 2:
            bgrs = MotionAugmentation.motion_affine(bgrs)
        if random.random() < self.prob_bgr_affine / 2:
            fgrs, phas, bgrs = MotionAugmentation.motion_affine(fgrs, phas, bgrs)

        # Still Affine
        if self.static_affine:
            fgrs, phas = MotionAugmentation.static_affine(fgrs, phas, scale_ranges=(0.5, 1))
            bgrs = MotionAugmentation.static_affine(bgrs, scale_ranges=(1, 1.5))

        # To tensor
        fgrs = torch.stack([F.to_tensor(fgr) for fgr in fgrs])
        phas = torch.stack([F.to_tensor(pha) for pha in phas])
        bgrs = torch.stack([F.to_tensor(bgr) for bgr in bgrs])
        bgrs_ = torch.stack([F.to_tensor(bgr) for bgr in bgrs_])

        # Resize
        if self.random_sized_crop:  # random crop of a square of size (self.size, self.size)
            square_size = (self.size, self.size)
            params = transforms.RandomResizedCrop.get_params(fgrs, scale=(1, 1), ratio=self.aspect_ratio_range)
            fgrs = F.resized_crop(fgrs, *params, square_size, interpolation=F.InterpolationMode.BILINEAR)
            phas = F.resized_crop(phas, *params, square_size, interpolation=F.InterpolationMode.BILINEAR)
            params = transforms.RandomResizedCrop.get_params(bgrs, scale=(1, 1), ratio=self.aspect_ratio_range)
            bgrs = F.resized_crop(bgrs, *params, square_size, interpolation=F.InterpolationMode.BILINEAR)
        else:  # resize such that smaller side has self.size
            # Note: We assume all bgrs and fgrs are horizontal
            # Most of the videos in DVM have the aspect ratio 16:9, but some are slightly different
            # We explicitly set the aspect ratio to 16:9 so they all have the same shape
            h, w = self.size, int(self.size * 16 / 9)
            bgrs = F.resize(bgrs, (h, w), interpolation=F.InterpolationMode.BILINEAR)
            bgrs_ = F.resize(bgrs_, (h, w), interpolation=F.InterpolationMode.BILINEAR)

            # Match size of fgrs to bgrs. Most fgrs are 432 x 768 - the standard aspect ratio of 16:9. A few are
            # 405 x 768, so we have to match the size to the bgr like this so they are the same size after resizing.
            fgrs = F.resize(fgrs, (h, w), interpolation=F.InterpolationMode.BILINEAR)
            phas = F.resize(phas, (h, w), interpolation=F.InterpolationMode.BILINEAR)

        # Horizontal flip
        if random.random() < self.prob_hflip:
            fgrs = F.hflip(fgrs)
            phas = F.hflip(phas)
        if random.random() < self.prob_hflip:
            bgrs = F.hflip(bgrs)

        # Noise
        if random.random() < self.prob_noise:
            fgrs, bgrs = MotionAugmentation.motion_noise(fgrs, bgrs)

        # Color jitter
        if random.random() < self.prob_color_jitter:
            fgrs = MotionAugmentation.motion_color_jitter(fgrs)
        if random.random() < self.prob_color_jitter:
            bgrs = MotionAugmentation.motion_color_jitter(bgrs)

        # Grayscale
        if random.random() < self.prob_grayscale:
            fgrs = F.rgb_to_grayscale(fgrs, num_output_channels=3).contiguous()
            bgrs = F.rgb_to_grayscale(bgrs, num_output_channels=3).contiguous()

        # Sharpen
        if random.random() < self.prob_sharpness:
            sharpness = random.random() * 8
            fgrs = F.adjust_sharpness(fgrs, sharpness)
            phas = F.adjust_sharpness(phas, sharpness)
            bgrs = F.adjust_sharpness(bgrs, sharpness)

        # Blur
        if random.random() < self.prob_blur / 3:
            fgrs, phas = MotionAugmentation.motion_blur(fgrs, phas)
        if random.random() < self.prob_blur / 3:
            bgrs = MotionAugmentation.motion_blur(bgrs)
        if random.random() < self.prob_blur / 3:
            fgrs, phas, bgrs = MotionAugmentation.motion_blur(fgrs, phas, bgrs)

        # Pause
        if random.random() < self.prob_pause:
            fgrs, phas, bgrs = MotionAugmentation.motion_pause(fgrs, phas, bgrs)

        return fgrs, phas, bgrs, bgrs_


class PrecapturedBgrAndPersonSameAugmentation:
    def __init__(self,
                 size,
                 prob_fgr_affine,
                 prob_bgr_affine,
                 prob_noise,
                 prob_color_jitter,
                 prob_grayscale,
                 prob_sharpness,
                 prob_blur,
                 prob_hflip,
                 prob_pause,
                 static_affine=True,
                 random_sized_crop=True,
                 aspect_ratio_range=(0.9, 1.1)):
        self.size = size
        self.prob_fgr_affine = prob_fgr_affine
        self.prob_bgr_affine = prob_bgr_affine
        self.prob_noise = prob_noise
        self.prob_color_jitter = prob_color_jitter
        self.prob_grayscale = prob_grayscale
        self.prob_sharpness = prob_sharpness
        self.prob_blur = prob_blur
        self.prob_hflip = prob_hflip
        self.prob_pause = prob_pause
        self.static_affine = static_affine
        self.aspect_ratio_range = aspect_ratio_range
        self.random_sized_crop = random_sized_crop

    def __call__(self, fgrs, phas, bgrs, bgrs_):
        # Foreground affine
        if random.random() < self.prob_fgr_affine:
            fgrs, phas = MotionAugmentation.motion_affine(fgrs, phas)

        # Background affine
        if random.random() < self.prob_bgr_affine / 2:
            bgrs, bgrs_ = MotionAugmentation.motion_affine(bgrs, bgrs_)
        if random.random() < self.prob_bgr_affine / 2:
            fgrs, phas, bgrs, bgrs_ = MotionAugmentation.motion_affine(fgrs, phas, bgrs, bgrs_)

        # Still Affine
        if self.static_affine:
            fgrs, phas = MotionAugmentation.static_affine(fgrs, phas, scale_ranges=(0.5, 1))
            bgrs, bgrs_ = MotionAugmentation.static_affine(bgrs, bgrs_, scale_ranges=(1, 1.5))

        # To tensor
        fgrs = torch.stack([F.to_tensor(fgr) for fgr in fgrs])
        phas = torch.stack([F.to_tensor(pha) for pha in phas])
        bgrs = torch.stack([F.to_tensor(bgr) for bgr in bgrs])
        bgrs_ = torch.stack([F.to_tensor(bgr) for bgr in bgrs_])

        # Resize
        if self.random_sized_crop:  # random crop of a square of size (self.size, self.size)
            square_size = (self.size, self.size)
            params = transforms.RandomResizedCrop.get_params(fgrs, scale=(1, 1), ratio=self.aspect_ratio_range)
            fgrs = F.resized_crop(fgrs, *params, square_size, interpolation=F.InterpolationMode.BILINEAR)
            phas = F.resized_crop(phas, *params, square_size, interpolation=F.InterpolationMode.BILINEAR)
            params = transforms.RandomResizedCrop.get_params(bgrs, scale=(1, 1), ratio=self.aspect_ratio_range)
            bgrs = F.resized_crop(bgrs, *params, square_size, interpolation=F.InterpolationMode.BILINEAR)
        else:  # resize such that smaller side has self.size
            # Note: We assume all bgrs and fgrs are horizontal
            # Most of the videos in DVM have the aspect ratio 16:9, but some are slightly different
            # We explicitly set the aspect ratio to 16:9 so they all have the same shape
            h, w = self.size, int(self.size * 16 / 9)
            bgrs = F.resize(bgrs, (h, w), interpolation=F.InterpolationMode.BILINEAR)
            bgrs_ = F.resize(bgrs_, (h, w), interpolation=F.InterpolationMode.BILINEAR)

            # Match size of fgrs to bgrs. Most fgrs are 432 x 768 - the standard aspect ratio of 16:9. A few are
            # 405 x 768, so we have to match the size to the bgr like this so they are the same size after resizing.
            fgrs = F.resize(fgrs, (h, w), interpolation=F.InterpolationMode.BILINEAR)
            phas = F.resize(phas, (h, w), interpolation=F.InterpolationMode.BILINEAR)

        # Horizontal flip
        if random.random() < self.prob_hflip:
            fgrs = F.hflip(fgrs)
            phas = F.hflip(phas)
        if random.random() < self.prob_hflip:
            bgrs = F.hflip(bgrs)
            bgrs_ = F.hflip(bgrs_)

        # Noise
        if random.random() < self.prob_noise:
            fgrs, bgrs, bgrs_ = MotionAugmentation.motion_noise(fgrs, bgrs, bgrs_)

        # Color jitter
        if random.random() < self.prob_color_jitter:
            fgrs = MotionAugmentation.motion_color_jitter(fgrs)
        if random.random() < self.prob_color_jitter:
            bgrs, bgrs_ = MotionAugmentation.motion_color_jitter(bgrs, bgrs_)

        # Grayscale
        if random.random() < self.prob_grayscale:
            fgrs = F.rgb_to_grayscale(fgrs, num_output_channels=3).contiguous()
            bgrs = F.rgb_to_grayscale(bgrs, num_output_channels=3).contiguous()
            bgrs_ = F.rgb_to_grayscale(bgrs_, num_output_channels=3).contiguous()

        # Sharpen
        if random.random() < self.prob_sharpness:
            sharpness = random.random() * 8
            fgrs = F.adjust_sharpness(fgrs, sharpness)
            phas = F.adjust_sharpness(phas, sharpness)
            bgrs = F.adjust_sharpness(bgrs, sharpness)
            bgrs_ = F.adjust_sharpness(bgrs_, sharpness)

        # Blur
        if random.random() < self.prob_blur / 3:
            fgrs, phas = MotionAugmentation.motion_blur(fgrs, phas)
        if random.random() < self.prob_blur / 3:
            bgrs, bgrs_ = MotionAugmentation.motion_blur(bgrs, bgrs_)
        if random.random() < self.prob_blur / 3:
            fgrs, phas, bgrs, bgrs_ = MotionAugmentation.motion_blur(fgrs, phas, bgrs, bgrs_)

        # Pause
        if random.random() < self.prob_pause:
            fgrs, phas, bgrs, bgrs_ = MotionAugmentation.motion_pause(fgrs, phas, bgrs, bgrs_)

        return fgrs, phas, bgrs, bgrs_