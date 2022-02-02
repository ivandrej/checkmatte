import os
import sys

import torch

sys.path.append("..")
from model import model_concat_bgr
from inference import convert_video
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


def inference(experiment_dir, load_model, output_type="png_sequence"):
    if load_model == "RVM":
        model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").eval().cuda()
    else:
        # TODO: Allow for RVM model too
        model = model_concat_bgr.MattingNetwork("mobilenetv3", bgr_integration=args.bgr_integration).eval().cuda()
        model.load_state_dict(torch.load(load_model))

    # For each sample (directory with frames) in experiment dir
    for sample_name in sorted(os.listdir(os.path.join(experiment_dir, "input"))):
        out_dir = os.path.join(experiment_dir, "out", sample_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Save output to video
        if output_type == 'video':
            convert_video(
                model,  # The loaded model, can be on any device (cpu or cuda).
                input_source=os.path.join(experiment_dir, "input", sample_name),
                # A video file or an image sequence directory.
                input_resize=None,  # [Optional] Resize the input (also the output).
                downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
                output_type='video',  # Choose "video" or "png_sequence"
                output_composition=os.path.join(out_dir, "com.mp4"),  # File path if video; directory path if png sequence.
                output_alpha=os.path.join(out_dir, "pha.mp4"),  # [Optional] Output the raw alpha prediction.
                output_foreground=os.path.join(out_dir, "fgr.mp4"),  # [Optional] Output the raw foreground prediction.
                output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
                seq_chunk=12,  # Process n frames at once for better parallelism.
                num_workers=1,  # Only for image sequence input. Reader threads.
                progress=True  # Print conversion progress.
            )
        else:  # Save output to image seq
            convert_video(
                model,  # The loaded model, can be on any device (cpu or cuda).
                input_source=os.path.join(experiment_dir, "input", sample_name),
                bgr_source=args.bgr_source,
                # A video file or an image sequence directory.
                input_resize=args.input_resize,  # [Optional] Resize the input (also the output).
                downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
                output_type='png_sequence',  # Choose "video" or "png_sequence"
                # output_composition=os.path.join(out_dir, "com"),  # File path if video; directory path if png sequence.
                output_alpha=os.path.join(out_dir, "pha"),  # [Optional] Output the raw alpha prediction.
                bgr_src_pairs=os.path.join(out_dir, "bgr_src"),
                # output_foreground=os.path.join(out_dir, "fgr"),  # [Optional] Output the raw foreground prediction.
                output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
                seq_chunk=1,  # Process n frames at once for better parallelism.
                num_workers=1,  # Only for image sequence input. Reader threads.
                progress=True  # Print conversion progress.
            )

        # Used this for replicating rvm evaluation
        # convert_video(
        #     model,  # The loaded model, can be on any device (cpu or cuda).
        #     input_source=os.path.join(args.experiment_dir, "input", sample_name, "com"),
        #     # A video file or an image sequence directory.
        #     input_resize=None,  # [Optional] Resize the input (also the output).
        #     downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
        #     output_type='png_sequence',  # Choose "video" or "png_sequence"
        #     output_composition=os.path.join(out_dir, "com"),  # File path if video; directory path if png sequence.
        #     output_alpha=os.path.join(out_dir, "pha"),  # [Optional] Output the raw alpha prediction.
        #     output_foreground=os.path.join(out_dir, "fgr"),  # [Optional] Output the raw foreground prediction.
        #     output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
        #     seq_chunk=12,  # Process n frames at once for better parallelism.
        #     num_workers=1,  # Only for image sequence input. Reader threads.
        #     progress=True  # Print conversion progress.
        # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, required=True)
    parser.add_argument('--bgr-source', type=str, default=None)
    # only relevant if --bgr-source is specified
    parser.add_argument('--bgr-integration', type=str, choices=['concat', 'attention'], default='attention')
    parser.add_argument('--load-model', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--output-type', type=str, default='png_sequence', required=False)
    args = parser.parse_args()

    inference(args.experiment_dir, args.load_model, args.output_type)
