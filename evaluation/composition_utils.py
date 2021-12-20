import os

import numpy as np
import pims
from PIL import Image
from tqdm import tqdm


# TODO: Add option for bgr img_seq

def fgr_img_seq_bgr_vid(bgrs, fgr_path, pha_path, out_dir, args):
    framenames = sorted(os.listdir(fgr_path))

    num_frames = min(args.num_frames, min(len(framenames), len(bgrs)))
    for t in tqdm(range(num_frames)):
        with Image.open(os.path.join(fgr_path, framenames[t])) as fgr, \
                Image.open(os.path.join(pha_path, framenames[t])) as pha:
            fgr = fgr.convert('RGB')
            pha = pha.convert('L')

            if args.resize is not None:
                fgr = fgr.resize(args.resize, Image.BILINEAR)
                pha = pha.resize(args.resize, Image.BILINEAR)

        bgr = Image.fromarray(bgrs[t])
        bgr = bgr.resize(fgr.size, Image.BILINEAR)

        pha = np.asarray(pha).astype(float)[:, :, None] / 255
        com = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(bgr) * (1 - pha)))
        com.save(os.path.join(out_dir, str(t).zfill(4) + '.png'))


"""
Inputs:
 - bgr_frames: n frames
 - fgr: m frames
 - pha: m frames
 - num_frames: how many frames to consider
"""


def fgr_vid_bgr_vid(bgr_frames, fgr_path, pha_path, out_dir, args):
    fgr_frames = pims.PyAVVideoReader(fgr_path)
    pha_frames = pims.PyAVVideoReader(pha_path)
    assert (len(fgr_frames) == len(pha_frames))

    num_frames = min(args.num_frames, min(len(fgr_frames), len(bgr_frames)))
    for t in tqdm(range(num_frames)):
        fgr = Image.fromarray(fgr_frames[t])
        pha = Image.fromarray(pha_frames[t])
        pha = pha.convert('L')

        if args.resize is not None:
            fgr = fgr.resize(args.resize, Image.BILINEAR)
            pha = pha.resize(args.resize, Image.BILINEAR)

        bgr = Image.fromarray(bgr_frames[t])
        bgr = bgr.resize(fgr.size, Image.BILINEAR)

        pha = np.asarray(pha).astype(float)[:, :, None] / 255
        com = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(bgr) * (1 - pha)))
        com.save(os.path.join(out_dir, str(t).zfill(4) + '.png'))


"""
Inputs:
 - bgr: 1 frame (image)
 - fgr: m frames
 - pha: m frames
 - num_frames: how many frames to consider
"""


def fgr_vid_bgr_img(bgr, fgr_path, pha_path, out_dir, args):
    fgr_frames = pims.PyAVVideoReader(fgr_path)
    pha_frames = pims.PyAVVideoReader(pha_path)
    assert (len(fgr_frames) == len(pha_frames))

    num_frames = min(args.num_frames, len(fgr_frames))
    for t in tqdm(range(num_frames)):
        fgr = Image.fromarray(fgr_frames[t])
        pha = Image.fromarray(pha_frames[t])
        pha = pha.convert('L')

        if args.resize is not None:
            fgr = fgr.resize(args.resize, Image.BILINEAR)
            pha = pha.resize(args.resize, Image.BILINEAR)
        bgr = bgr.resize(fgr.size, Image.BILINEAR)

        pha = np.asarray(pha).astype(float)[:, :, None] / 255
        com = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(bgr) * (1 - pha)))
        com.save(os.path.join(out_dir, str(t).zfill(4) + '.png'))
