"""
    Same as run_inference.py, but it can perform inference on multiple epochs, e.g by passing --epochs 27 38
"""
import os

import torch

from inference import convert_video, FixedOffsetMatcher, get_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-source', type=str, required=True)
parser.add_argument('--bgr-source', type=str, default=None)
# only relevant if --bgr-source is specified
parser.add_argument('--model-type', type=str, choices=['addition', 'concat', 'f3'], default='addition')
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--load-model', type=str, required=True)
parser.add_argument('--temporal-offset', type=int, default=0)
parser.add_argument('--output-type', type=str, default='video', required=False)
parser.add_argument('--resize', type=int, default=(512, 288), nargs=2)
parser.add_argument('--epochs', '--list', nargs='+', required=True)
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

for epoch in args.epochs:
    out_dir = os.path.join(args.out_dir, f"epoch-{epoch}")

    model = get_model(args.model_type).eval().cuda()
    model.load_state_dict(torch.load(os.path.join(args.load_model, f"epoch-{epoch}.pth")))
    matcher = FixedOffsetMatcher(args.temporal_offset)

    if args.output_type == 'video':
        convert_video(
            model,  # The loaded model, can be on any device (cpu or cuda).
            input_source=args.input_source,
            input_resize=args.resize,  # [Optional] Resize the input (also the output).
            downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
            output_type='video',  # Choose "video" or "png_sequence"
            output_composition=f"{out_dir}/com.mp4",
            # File path if video; directory path if png sequence.
            output_alpha=f"{out_dir}/pha.mp4",  # [Optional] Output the raw alpha prediction.
            output_foreground=f"{out_dir}/fgr.mp4",
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
            matcher=matcher,
            bgr_source=args.bgr_source,
            input_resize=args.resize,  # [Optional] Resize the input (also the output).
            downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
            output_type="png_sequence",  # Choose "video" or "png_sequence"
            # output_composition=f"{out_dir}/com",
            output_alpha=f"{out_dir}/pha",  # [Optional] Output the raw alpha prediction.
            bgr_src_pairs=f"{out_dir}/bgr_src",
            # output_foreground=f"{out_dir}/fgr",
            # [Optional] Output the raw foreground prediction.
            seq_chunk=12,  # Process n frames at once for better parallelism.
            progress=True  # Print conversion progress.
        )
