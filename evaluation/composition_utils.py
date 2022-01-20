import os

import numpy as np
import pims
from PIL import Image
from tqdm import tqdm

def fgr_img_seq_bgr_vid(bgr_path, fgr_path, pha_path, out_dir, args):
    framenames = sorted(os.listdir(fgr_path))

    if os.path.isdir(bgr_path):  # img seq
        bgr_framenames = sorted(os.listdir(bgr_path))
        bgrs = [os.path.join(bgr_path, framename) for framename in bgr_framenames]
    elif bgr_path.endswith(".mp4") or bgr_path.endswith(".MTS"):  # video background
        bgrs = pims.PyAVVideoReader(bgr_path)
    else:  # TODO: Add option for still image
        raise Exception("Single image bgr not supported")

    num_frames = min(args.num_frames, min(len(framenames), len(bgrs)))
    for t in tqdm(range(num_frames)):
        with Image.open(os.path.join(fgr_path, framenames[t])) as fgr, \
                Image.open(os.path.join(pha_path, framenames[t])) as pha:
            fgr = fgr.convert('RGB')
            pha = pha.convert('L')

            if args.resize is not None:
                fgr = fgr.resize(args.resize, Image.BILINEAR)
                pha = pha.resize(args.resize, Image.BILINEAR)

        bgr = get_bgr_frame(bgrs, t)
        # bgr = Image.fromarray(bgrs[t])
        bgr = bgr.resize(fgr.size, Image.BILINEAR)

        pha = np.asarray(pha).astype(float)[:, :, None] / 255
        com = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(bgr) * (1 - pha)))
        com.save(os.path.join(out_dir, str(t).zfill(4) + '.png'))


"""
    Allows to handle video and img seq bgrs together
    - if img_seq, bgrs is a list of full paths to frames
    - if vid, bgrs is a video
"""
def get_bgr_frame(bgrs, t):
    if type(bgrs) is list:  # img seq
        with Image.open(bgrs[t]) as bgr:
            bgr = bgr.convert('RGB')
            return bgr
    else:
        return Image.fromarray(bgrs[t])



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
