import os
import sys

import torch

sys.path.append("..")
from model.model import MattingNetwork
from rvm_inference import convert_video
import argparse

"""
Perform inference on all samples in an experiment dir. 
Save output in `/out` in each of the sample dirs
File structure:
 - experiment1
   - input
      - 0000
         - 0000.png
         - 0002.png
          ...
      - ....
    - out
      - 0000
        - com.mp4
        - pha.mp4
        - fgr.mp4
"""

"""
    load_model: the path to a .pth RVM checkpoint. If none, will load the fully trained RVM model
"""
def inference(experiment_dir, input_dir, input_resize, load_model=None):
    model = MattingNetwork("mobilenetv3").eval().cuda()
    if load_model is None:
        model.load_state_dict(torch.load("/media/andivanov/DATA/training/rvm_mobilenetv3.pth"))
    else:
        model.load_state_dict(torch.load(load_model))

    if input_dir is None:
        input_dir = os.path.join(experiment_dir, "input")

    # For each sample (directory with frames) in experiment dir
    for sample_name in sorted(os.listdir(input_dir)):
        out_dir = os.path.join(experiment_dir, "out", sample_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Save output to image seq
        convert_video(
            model,  # The loaded model, can be on any device (cpu or cuda).
            input_source=os.path.join(input_dir, sample_name),
            # A video file or an image sequence directory.
            input_resize=input_resize,  # [Optional] Resize the input (also the output).
            downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
            output_type='png_sequence',  # Choose "video" or "png_sequence"
            # output_composition=os.path.join(out_dir, "com"),  # File path if video; directory path if png sequence.
            output_alpha=os.path.join(out_dir, "pha"),  # [Optional] Output the raw alpha prediction.
            # output_foreground=os.path.join(out_dir, "fgr"),  # [Optional] Output the raw foreground prediction.
            output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
            seq_chunk=12,  # Process n frames at once for better parallelism.
            num_workers=1,  # Only for image sequence input. Reader threads.
            progress=True  # Print conversion progress.
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, required=True)
    # If not specified, input_dir is experiment_dir/input
    parser.add_argument('--input-dir', type=str, default=None)
    # only relevant if --bgr-source is specified
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--output-type', type=str, default='png_sequence', required=False)
    args = parser.parse_args()

    inference(args.experiment_dir, args.input_dir, args.input_resize)
