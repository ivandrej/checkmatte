"""

"""
import os

import torch

from model import model, model_concat_bgr

from inference import convert_video

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-source', type=str, required=True)
parser.add_argument('--bgr-source', type=str, default=None)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--load-model', type=str, required=True)
parser.add_argument('--output-type', type=str, default='video', required=False)
parser.add_argument('--resize', type=int, default=(512, 288), nargs=2)
args = parser.parse_args()

if args.load_model == "RVM":
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").eval().cuda()
else:
    if args.bgr_source:
        # TODO: Better naming of models
        model = model_concat_bgr.MattingNetwork("mobilenetv3").eval().cuda()
    else:
        model = model.MattingNetwork("mobilenetv3").eval().cuda()
    model.load_state_dict(torch.load(args.load_model))

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

if args.output_type == 'video':
    convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=args.input_source,
        bgr_source=args.bgr_source,
        input_resize=args.resize,  # [Optional] Resize the input (also the output).
        downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
        output_type='video',  # Choose "video" or "png_sequence"
        output_composition=f"{args.out_dir}/com.mp4",
        # File path if video; directory path if png sequence.
        output_alpha=f"{args.out_dir}/pha.mp4",  # [Optional] Output the raw alpha prediction.
        output_foreground=f"{args.out_dir}/fgr.mp4",
        # [Optional] Output the raw foreground prediction.
        output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
        seq_chunk=12,  # Process n frames at once for better parallelism.
        num_workers=1,  # Only for image sequence input. Reader threads.
        progress=True  # Print conversion progress.
    )
else:  # save as png seq
    convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=args.input_source,
        bgr_source=args.bgr_source,
        input_resize=args.resize,  # [Optional] Resize the input (also the output).
        downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
        output_type="png_sequence",  # Choose "video" or "png_sequence"
        output_composition=f"{args.out_dir}/com",
        output_alpha=f"{args.out_dir}/pha",  # [Optional] Output the raw alpha prediction.
        output_foreground=f"{args.out_dir}/fgr",
        # [Optional] Output the raw foreground prediction.
        # TODO: Make this an argument
        seq_chunk=1,  # Process n frames at once for better parallelism.
        progress=True  # Print conversion progress.
    )
