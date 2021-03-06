import argparse
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

from inference import get_model


def varying_temporaloffset_experiment(model,
                                      input_source: str,
                                      bgr_source: str,
                                      output_dir: str,
                                      input_resize: Optional[Tuple[int, int]] = None,
                                      bgr_rotation: Optional[Tuple[int, int]] = (0, 0),
                                      attention_visualizer=None):
    # Initialize transform
    if input_resize is not None:
        transform_src = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform_src = transforms.ToTensor()

    if input_resize is not None:
        transform_bgr = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.RandomRotation(bgr_rotation, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
    else:
        transform_bgr = transforms.ToTensor()

    # Read all bgr frames in memory
    bgrs = []
    for frame in sorted(os.listdir(bgr_source)):
        with Image.open(os.path.join(bgr_source, frame)) as bgr:
            bgr.load()
        if transform_src is not None:
            bgr = transform_bgr(bgr)
        bgrs.append(bgr)

    person_frames = []
    for frame in sorted(os.listdir(input_source)):
        with Image.open(os.path.join(input_source, frame)) as pframe:
            pframe.load()
        if transform_src is not None:
            pframe = transform_src(pframe)
        person_frames.append(pframe)

    pha_dir = os.path.join(output_dir, 'pha')
    os.makedirs(pha_dir, exist_ok=True)
    side_by_side_dir = os.path.join(output_dir, 'side_by_side')
    os.makedirs(side_by_side_dir, exist_ok=True)

    # Inference
    model = get_model("f3").eval().cuda()
    model.load_state_dict(torch.load(args.load_model))
    model = model.eval()
    param = next(model.parameters())
    dtype = param.dtype
    device = param.device

    target_t = 97
    T = 15
    with torch.no_grad():
        for bgr_t in tqdm(range(len(bgrs))):
            # construct the T = 15 sequence p[t - 15, t]
            src = []
            for person_t in range(target_t - T + 1, target_t + 1):
                src.append(person_frames[person_t])
            src = torch.stack(src)  # List of T [C, H, W] to tensor of [T, C, H, W]

            bgr = bgrs[bgr_t].repeat(src.shape[0], 1, 1, 1)
            src = src.to(device, dtype, non_blocking=True).unsqueeze(0)  # [1, T, C, H, W]
            bgr = bgr.to(device, dtype, non_blocking=True).unsqueeze(0)  # [1, T, C, H, W]

            return_intermediate = (attention_visualizer is not None)
            rec = [None] * 4
            fgr, pha, attention, *rec = model(src, bgr, *rec,
                                              downsample_ratio=1, return_intermediate=return_intermediate)

            target_pha = pha[0][-1].detach().cpu().mul(255).byte().permute(1, 2, 0).numpy()
            target_pha = cv2.cvtColor(target_pha, cv2.COLOR_GRAY2RGB)
            # print("Target pha shape: ", target_pha.shape)
            target_src = src[0][-1].detach().cpu().mul(255).permute(1, 2, 0).numpy()
            bgr = bgr[0][0].detach().cpu().mul(255).permute(1, 2, 0).numpy()

            res = np.uint8(np.concatenate((target_pha, bgr, target_src), axis=1))
            # print("Res max and min: ", np.max(res), np.min(res))
            # print("Res shape: ", res.shape)
            Image.fromarray(res).save(os.path.join(side_by_side_dir, str(bgr_t).zfill(4) + '.png'))
            to_pil_image(pha[0][-1]).save(os.path.join(output_dir, str(bgr_t).zfill(4) + '.png'))


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, required=True)
    parser.add_argument('--person-source', type=str, required=True)
    parser.add_argument('--bgr-source', type=str, required=True)
    parser.add_argument('--load-model', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    args = parser.parse_args()

    model = get_model("f3").eval().cuda()
    model.load_state_dict(torch.load(args.load_model))

    varying_temporaloffset_experiment(model, args.person_source,
                                      args.bgr_source, args.experiment_dir,
                                      input_resize=args.input_resize)
