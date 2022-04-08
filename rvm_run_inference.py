import os
import torch
import argparse

from model import rvm
from rvm_inference import convert_video


parser = argparse.ArgumentParser()
parser.add_argument('--input-source', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--resize', type=int, default=None, nargs=2)
parser.add_argument('--downsample_ratio', type=float, default=None)
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

model = model.MattingNetwork("mobilenetv3").eval().cuda()
model.load_state_dict(torch.load("/media/andivanov/DATA/training/rvm_mobilenetv3.pth"))
out_dir = os.path.join(args.out_dir)

convert_video(
    model,  # The loaded model, can be on any device (cpu or cuda).
    input_source=args.input_source,
    input_resize=args.resize,  # [Optional] Resize the input (also the output).
    downsample_ratio=args.downsample_ratio,  # [Optional] If None, make downsampled max size be 512px.
    output_type="png_sequence",  # Choose "video" or "png_sequence"
    output_composition=f"{out_dir}/com",
    output_alpha=f"{out_dir}/pha",  # [Optional] Output the raw alpha prediction.
    output_foreground=f"{out_dir}/fgr",
    # [Optional] Output the raw foreground prediction.
    seq_chunk=12,  # Process n frames at once for better parallelism.
    progress=True  # Print conversion progress.
)
