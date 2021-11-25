"""
python generate_videomatte_with_background_video.py \
    --videomatte-dir ../matting-data/VideoMatte240K_JPEG_HD/test \
    --background-dir ../matting-data/BackgroundVideos_mp4/test \
    --resize 512 288 \
    --out-dir ../matting-data/evaluation/vidematte_motion_sd/
"""

import argparse
import os
import pims
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

bgrs_per_fgr = 3

# Foreground is an img sequence
# NOT WORKING RN
def fgr_img_seq():
    # This dir needs to contain fgr and pha
    fgr_dir = "/home/andivanov/dev/data/RVM_videomatte_evaluation_1920x1080/videomatte_motion/0015" 
    framenames = sorted(os.listdir(os.path.join(fgr_dir, "fgr")))
    
    num_frames = min(args.num_frames, len(framenames))
    for t in tqdm(range(num_frames)):
        with Image.open(os.path.join(fgr_dir, 'fgr', framenames[base_t + t])) as fgr, \
                Image.open(os.path.join(fgr_dir, 'pha', framenames[base_t + t])) as pha:
            fgr = fgr.convert('RGB')
            pha = pha.convert('L')
            
            if args.resize is not None:
                fgr = fgr.resize(args.resize, Image.BILINEAR)
                pha = pha.resize(args.resize, Image.BILINEAR)
                
        bgr = Image.fromarray(bgrs[t])
        bgr = bgr.resize(fgr.size, Image.BILINEAR)
        
        pha = np.asarray(pha).astype(float)[:, :, None] / 255
        com = Image.fromarray(np.uint8(np.asarray(fgr) * pha + np.asarray(bgr) * (1 - pha)))
        com.save(os.path.join(out_dir, 'com', str(t).zfill(4) + '.png'))

"""
Inputs:
 - bgr_frames: n frames
 - fgr: m frames
 - pha: m frames
 - num_frames: how many frames to consider
"""
def fgr_vid_bgr_vid(bgr_frames, fgr_path, pha_path, out_dir):
    fgr_frames = pims.PyAVVideoReader(fgr_path)
    pha_frames = pims.PyAVVideoReader(pha_path)
    assert(len(fgr_frames) == len(pha_frames))

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
def fgr_vid_bgr_img(bgr, fgr_path, pha_path, out_dir):
    fgr_frames = pims.PyAVVideoReader(fgr_path)
    pha_frames = pims.PyAVVideoReader(pha_path)
    assert(len(fgr_frames) == len(pha_frames))

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


"""
Composite multiple foregrounds to multiple backgrounds.
Right now for each foreground we have 3 backgrounds 
""" 
def composite_multiple_bgr():
    print(len(bgr_paths), len(fgr_paths), len(pha_paths))
    assert (len(bgr_paths) == bgrs_per_fgr * len(fgr_paths) == bgrs_per_fgr * len(pha_paths)) 
    for i, (fgr_path, pha_path) in enumerate(zip(fgr_paths, pha_paths)):
        for bgr_path in bgr_paths[i * bgrs_per_fgr: (i+1) * bgrs_per_fgr]:   
            out_dir = os.path.join(args.out_dir, "{}_{}".format(
                clipname_from_path(pha_path), clipname_from_path(bgr_path)))
            os.makedirs(out_dir)
            if bgr_path.endswith(".mp4") or bgr_path.endswith(".MTS") : # video background
                bgr_frames = pims.PyAVVideoReader(bgr_path) 
                fgr_vid_bgr_vid(bgr_frames, fgr_path, pha_path, out_dir)
            else: # image (static) background
                with Image.open(bgr_path) as bgr:
                    bgr = bgr.convert('RGB')
                    fgr_vid_bgr_img(bgr, fgr_path, pha_path, out_dir)


def clipname_from_path(path):
    from pathlib import Path
    return Path(path).stem
            

# Triplets of very dynamic, medium dynamic and static 
bgr_paths = [
        "/home/andivanov/dev/data/dynamic_backgrounds_youtube/car_1.mp4",
        "/data/DVM/bg/test/0180.mp4",
        "/home/andivanov/dev/data/BGM_Image_Backgrounds/classroom_interior_318.br420-1.jpg",

        "/home/andivanov/dev/data/dynamic_backgrounds_youtube/nature_1.1.mp4",
        "/data/DVM/bg/test/0010.mp4",
        "/home/andivanov/dev/data/BGM_Image_Backgrounds/bar_interior_348.death&co__mg_9751-copy.jpg",

        "/home/andivanov/dev/data/dynamic_backgrounds_youtube/nature_2.1.mp4",
        "/data/DVM/bg/test/0010.mp4",
        "/home/andivanov/dev/data/BGM_Image_Backgrounds/canyon_0369.jpg",

        "/data/our_dynamic_foreground_captures/Captures Andrej/00037.MTS",
        "/data/DVM/bg/test/0015.mp4",
        "/home/andivanov/dev/data/BGM_Image_Backgrounds/garage_interior_121.9.-2019-e-3rd-interior-departamento-1.jpg",

        "/data/our_dynamic_foreground_captures/Captures Andrej/00044.MTS",
        "/data/DVM/bg/test/0043.mp4",
        "/home/andivanov/dev/data/BGM_Image_Backgrounds/empty_city_115.empty-city-streets-atlanta-georgia-jackson-bridge.jpg"
]
fgr_paths = [
    "/home/andivanov/dev/data/VideoMatte240K/test/fgr/0004.mp4",
    "/home/andivanov/dev/data/VideoMatte240K/test/fgr/0003.mp4",
    "/home/andivanov/dev/data/VideoMatte240K/test/fgr/0002.mp4",
    "/home/andivanov/dev/data/VideoMatte240K/test/fgr/0001.mp4",
    "/home/andivanov/dev/data/VideoMatte240K/test/fgr/0000.mp4",
]
pha_paths = [
    "/home/andivanov/dev/data/VideoMatte240K/test/pha/0004.mp4",
    "/home/andivanov/dev/data/VideoMatte240K/test/pha/0003.mp4",
    "/home/andivanov/dev/data/VideoMatte240K/test/pha/0002.mp4",
    "/home/andivanov/dev/data/VideoMatte240K/test/pha/0001.mp4",
    "/home/andivanov/dev/data/VideoMatte240K/test/pha/0000.mp4",    
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--videomatte-dir', type=str, required=True)
    # parser.add_argument('--background-dir', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--num-frames', type=int, default=100)
    parser.add_argument('--resize', type=int, default=None, nargs=2)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir), exist_ok=True)

    # Helps in cropping exactly args.num_frames from the foreground video. We acheive this later by starting at base_t
    # base_t = random.choice(range(len(framenames) - args.num_frames))
    base_t = 0

    composite_multiple_bgr()