"""
    Reads a directory of person frames and a directory of bgr frames, and outputs alpha mattes in a directory for results

    Example usage:
    To run inference on the F3 resolution 227x128 model:
    python run_inference.py --input-source /person_frames --bgr-source /bgr_frames --out-dir /result \
    --load-model checkpoint/stage1/epoch38.pth --resize 227 128 --model f3

"""
import argparse
import os

import torch

from inference import convert_video, FixedOffsetMatcher, get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # a directory of all frames for the person input video, labelled 0000, 0001, ...
    parser.add_argument('--input-source', type=str, required=True)
    # a directory of all frames for the background input video, labelled 0000, 0001, ...
    parser.add_argument('--bgr-source', type=str, default=None)
    # only relevant if --bgr-source is specified
    parser.add_argument('--model-type', type=str, choices=['addition', 'concat', 'f3'], default='f3')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--load-model', type=str, required=True)
    parser.add_argument('--temporal-offset', type=int, default=0)
    parser.add_argument('--output-type', type=str, default='png_sequence')
    parser.add_argument('--resize', type=int, default=(512, 288), nargs=2)
    args = parser.parse_args()

    model = get_model(args.model_type).eval().cuda()
    model.load_state_dict(torch.load(os.path.join(args.load_model)))
    matcher = FixedOffsetMatcher(args.temporal_offset)

    os.makedirs(args.out_dir, exist_ok=True)

   # save as png seq
    convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=args.input_source,
        matcher=matcher,
        bgr_source=args.bgr_source,
        input_resize=args.resize,  # [Optional] Resize the input (also the output).
        downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
        output_type="png_sequence",  # Choose "video" or "png_sequence"
        # output_composition=f"{out_dir}/com",
        output_alpha=f"{args.out_dir}/pha",  # [Optional] Output the raw alpha prediction.
        bgr_src_pairs=f"{args.out_dir}/bgr_src",
        # output_foreground=f"{out_dir}/fgr",
        # [Optional] Output the raw foreground prediction.
        seq_chunk=12,  # Process n frames at once for better parallelism.
        progress=True  # Print conversion progress.
    )

