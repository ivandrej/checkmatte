"""
python python composite.py
    --out-dir ~/dev/data/composited_evaluation/experiment3/input
    --resize 512 288
    --num-frames 600
"""

import argparse
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
    return "{}_{}".format(clipname_from_path(pha_path), clipname_from_path(bgr_path))


bgr_paths = {
    "dynamic": [
        "/media/andivanov/DATA/dynamic_backgrounds_captured/construction_site_1.mp4",
        "/media/andivanov/DATA/dynamic_backgrounds_captured/stairs.mp4",
        "/media/andivanov/DATA/dynamic_backgrounds_captured/yard.mp4",
        "/media/andivanov/DATA/dynamic_backgrounds_captured/bikes_2.mp4",
        "/data/our_dynamic_foreground_captures/Captures Andrej/00037.MTS",
        "/data/our_dynamic_foreground_captures/Captures Andrej/00044.MTS"
    ],
    "semi_dynamic": [
        "/media/andivanov/DATA/DVM/bg/test/0065.mp4",
        "/media/andivanov/DATA/DVM/bg/test/0073.mp4",
        "/media/andivanov/DATA/DVM/bg/test/0010.mp4",
        "/media/andivanov/DATA/DVM/bg/test/0015.mp4",
        "/media/andivanov/DATA/DVM/bg/test/0043.mp4",
        "/media/andivanov/DATA/DVM/bg/test/0070.mp4"
    ],
    "static": [
        "/home/andivanov/dev/data/BGM_Image_Backgrounds/classroom_interior_318.br420-1.jpg",
        "/home/andivanov/dev/data/BGM_Image_Backgrounds/bar_interior_348.death&co__mg_9751-copy.jpg",
        "/home/andivanov/dev/data/BGM_Image_Backgrounds/canyon_0369.jpg",
        "/home/andivanov/dev/data/BGM_Image_Backgrounds/garage_interior_121.9.-2019-e-3rd-interior-departamento-1.jpg",
        "/home/andivanov/dev/data/BGM_Image_Backgrounds/church_interior_226.res-church.jpg",
        "/home/andivanov/dev/data/BGM_Image_Backgrounds/empty_city_115.empty-city-streets-atlanta-georgia-jackson-bridge.jpg"
    ]}
def clipname_from_path(path):
    from pathlib import Path
    return Path(path).stem


# TODO: Move this to a metadata file

fgr_paths = [
    "/media/andivanov/DATA/VideoMatte240K/test/fgr/0004.mp4",
    "/media/andivanov/DATA/VideoMatte240K/test/fgr/0003.mp4",
    "/media/andivanov/DATA/VideoMatte240K/test/fgr/0002.mp4",
    "/media/andivanov/DATA/VideoMatte240K/test/fgr/0001.mp4",
    "/media/andivanov/DATA/VideoMatte240K/test/fgr/0000.mp4",
]
pha_paths = [
    "/media/andivanov/DATA/VideoMatte240K/test/pha/0004.mp4",
    "/media/andivanov/DATA/VideoMatte240K/test/pha/0003.mp4",
    "/media/andivanov/DATA/VideoMatte240K/test/pha/0002.mp4",
    "/media/andivanov/DATA/VideoMatte240K/test/pha/0001.mp4",
    "/media/andivanov/DATA/VideoMatte240K/test/pha/0000.mp4",
]

"""
    Contains paths to fgr, pha and bgr we want to composite
"""


class CompositedClipPaths:
    def __init__(self, fgr_path, pha_path, bgr_path, bgr_type):
        self.fgr_path = fgr_path
        self.pha_path = pha_path
        self.bgr_path = bgr_path
        self.clipname = com_clipname(pha_path, bgr_path)
        self.bgr_type = bgr_type


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
