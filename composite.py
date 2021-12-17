"""
python python composite.py
    --out-dir ~/dev/data/composited_evaluation/experiment3/input
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

from composition_utils import fgr_vid_bgr_vid, fgr_vid_bgr_img

bgrs_per_fgr = 3

"""
Composite a list of foreground, background pairs
"""
def composite_multiple_bgr(clips):
    print("Will create {} composed clips".format(len(clips)))
    for com_paths in clips:
        print(com_paths.bgr_path)
        out_dir = os.path.join(args.out_dir, com_paths.clipname)
        print("Creating: ", com_paths.clipname)
        os.makedirs(out_dir, exist_ok=True)
        if com_paths.bgr_path.endswith(".mp4") or com_paths.bgr_path.endswith(".MTS"):  # video background
            bgr_frames = pims.PyAVVideoReader(com_paths.bgr_path)
            fgr_vid_bgr_vid(bgr_frames, com_paths.fgr_path, com_paths.pha_path, out_dir, args)
        else:  # image (static) background
            with Image.open(com_paths.bgr_path) as bgr:
                bgr = bgr.convert('RGB')
                fgr_vid_bgr_img(bgr, com_paths.fgr_path, com_paths.pha_path, out_dir, args)


def com_clipname(pha_path, bgr_path):
    # print("Pha path", pha_path)
    # if os.path.isdir(pha_path):
    #     print("Pha is dir")
    #     fgr_clip = clipname_from_path(os.path.join(pha_path, ".."))
    # else:
    #     fgr_clip = clipname_from_path(pha_path)
    #
    # print("Fgr clip: ", fgr_clip)
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

# TODO: pass in json file as cmdl argument
# TODO: Move to a separate file (data reader)
# with open("experiment_metadata/VideoMatte5x18_img_seq.json", "r") as f:
with open("experiment_metadata/VideoMatte5x18_img_seq.json", "r") as f:
    data = json.load(f)
    bgr_paths = data["bgr_paths"]
    fgr_paths = data["fgr_paths"]
    pha_paths = data["pha_paths"]

assert(len(fgr_paths) == len(pha_paths))

# Match foregrounds with backgrounds. Produces a flat list of (fgr, pha, bgr)
# Composite each fgr onto each bgr
bgr_triples = list(zip(bgr_paths["dynamic"], bgr_paths["semi_dynamic"], bgr_paths["static"]))
clips: List[CompositedClipPaths] = []
print("{} foregrounds, {} backgrounds: ".format(len(fgr_paths), len(bgr_triples) * 3))
for i, (fgr_path, pha_path) in enumerate(zip(fgr_paths, pha_paths)):
    for dynamic_bgr, semi_dynamic_bgr, static_bgr in bgr_triples:
        for video_path in (dynamic_bgr, semi_dynamic_bgr, static_bgr, fgr_path, pha_path):
            if not os.path.exists(video_path):
                raise Exception("{} does not exists".format(video_path))
        dynamic_com = CompositedClipPaths(fgr_path, pha_path, dynamic_bgr, "dynamic")
        semi_dynamic_com = CompositedClipPaths(fgr_path, pha_path, semi_dynamic_bgr, "semi_dynamic")
        static_com = CompositedClipPaths(fgr_path, pha_path, static_bgr, "static")

        clips.append(dynamic_com)
        clips.append(semi_dynamic_com)
        clips.append(static_com)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--num-frames', type=int, default=100)
    parser.add_argument('--resize', type=int, default=None, nargs=2)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir), exist_ok=True)

    # Helps in cropping exactly args.num_frames from the foreground video. We acheive this later by starting at base_t
    # base_t = random.choice(range(len(framenames) - args.num_frames))
    base_t = 0

    composite_multiple_bgr(clips)
