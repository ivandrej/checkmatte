"""
python composite.py
    --out-dir ~/dev/data/composited_evaluation/experiment3/input
    --experiment-metadata experiment_metadata/VMxDVM_0013.json
    --resize 512 288
    --num-frames 600
"""

import argparse
import json
import os
from typing import List

import numpy as np
import pims
from PIL import Image
from tqdm import tqdm

from composition_utils import fgr_vid_bgr_vid, fgr_vid_bgr_img, fgr_img_seq_bgr_vid

bgrs_per_fgr = 3

"""
Composite a list of foreground, background pairs
"""


def composite_fgrs_to_bgrs(base_out_dir, experiment_metadata, args):
    os.makedirs(base_out_dir, exist_ok=True)

    clips = read_metadata(experiment_metadata)
    print("Will create {} composed clips".format(len(clips)))

    for com_paths in clips:
        print(com_paths.bgr_path)
        out_dir = os.path.join(base_out_dir, com_paths.clipname)
        print("Creating: ", com_paths.clipname)
        os.makedirs(out_dir, exist_ok=True)

        if os.path.isdir(com_paths.fgr_path):  # Fgr is img seq
            fgr_img_seq_bgr_vid(com_paths.bgr_path, com_paths.fgr_path, com_paths.pha_path, out_dir, args)
        else:   # Fgr is video
            # TODO: Fix fgr video
            raise Exception("Fgr video is broken")
            # fgr_vid_bgr_img(bgr, com_paths.fgr_path, com_paths.pha_path, out_dir, args)


def com_clipname(pha_path, bgr_path):
    return "{}_{}".format(clipname_from_path(pha_path), clipname_from_path(bgr_path))


def clipname_from_path(path):
    from pathlib import Path
    return Path(path).stem


class CompositedClipPaths:
    def __init__(self, fgr_path, pha_path, bgr_path, bgr_type):
        self.fgr_path = fgr_path
        self.pha_path = pha_path
        self.bgr_path = bgr_path
        self.clipname = com_clipname(pha_path, bgr_path)
        self.bgr_type = bgr_type


def read_metadata(experiment_metadata_path):
    with open(experiment_metadata_path, "r") as f:
        data = json.load(f)
        bgr_paths = data["bgr_paths"]
        fgr_paths = data["fgr_paths"]
        pha_paths = data["pha_paths"]

    assert (len(fgr_paths) == len(pha_paths))

    # Match foregrounds with backgrounds. Produces a flat list of (fgr, pha, bgr)
    # Composite each fgr onto each bgr
    bgrs = []
    bgr_types = []
    for bgr_type, bgrs_for_type in bgr_paths.items():
        bgrs = bgrs + bgrs_for_type
        bgr_types = bgr_types + len(bgrs_for_type) * [bgr_type]

    clips: List[CompositedClipPaths] = []
    print("{} foregrounds, {} backgrounds: ".format(len(fgr_paths), len(bgrs)))
    for i, (fgr_path, pha_path) in enumerate(zip(fgr_paths, pha_paths)):
        for bgr, bgr_type in zip(bgrs, bgr_types):
            for video_path in (bgr, fgr_path, pha_path):
                if not os.path.exists(video_path):
                    raise Exception("{} does not exists".format(video_path))
            com = CompositedClipPaths(fgr_path, pha_path, bgr, bgr_type)
            clips.append(com)

    return clips


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-metadata', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--num-frames', type=int, default=100)
    parser.add_argument('--resize', type=int, default=None, nargs=2)
    args = parser.parse_args()

    composite_fgrs_to_bgrs(args.out_dir, args.experiment_metadata, args)
