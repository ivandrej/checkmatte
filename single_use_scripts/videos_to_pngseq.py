import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--outdir', type=str,
                    default='/media/andivanov/DATA/dynamic_backgrounds_captured/real_test_videos/')
args = parser.parse_args()

def to_pngseq(clipname, tag):
    """
    :param clipname: Name of video
    :param tag: 'person' or 'bgr'
    """
    clipgroup = args.dir.split('/')[-1]
    vidpath = os.path.join(args.dir, clipname, f'{tag}.mp4')
    outdir = os.path.join(args.outdir, clipgroup, clipname, tag)
    os.makedirs(outdir, exist_ok=True)
    os.system(f"ffmpeg -i {vidpath} {outdir}/%04d.png")

for clipname in os.listdir(args.dir):
    to_pngseq(clipname, 'person')
    to_pngseq(clipname, 'bgr')
