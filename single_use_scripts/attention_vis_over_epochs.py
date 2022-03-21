"""

"""
import os
import sys

import torch

sys.path.append('..')
from model import model, model_attention_addition, model_attention_concat

from inference_write_single_frame import convert_video, FixedOffsetMatcher

import argparse

from visualization import visualize_attention


def get_spatial_locations(H, W):
    return [
        (H // 2, W // 2),  # center
        (3 * H // 4, W // 2),  # around feet
        (3 * H // 4, W // 4),  # bottom left corner
        (3 * H // 4, 3 * W // 4)  # bottom right corner
    ]


frameids = [0, 10, 60, 100]


class Visualizer:
    def __init__(self, epoch):
        self.outdir = os.path.join(args.out_dir, "attention_vis")
        self.counter = 0
        self.epoch = epoch

    def __call__(self, attention):
        assert (attention.dim() == 5)
        T, H, W, _, _ = attention.shape
        spatial_locations = get_spatial_locations(H, W)
        for t in range(attention.shape[0]):
            if self.counter in frameids:
                for h, w in spatial_locations:
                    outdir = os.path.join(self.outdir, f"{h}-{w}-{self.counter}")
                    os.makedirs(outdir, exist_ok=True)
                    fig = visualize_attention.get_attention_fig(attention, h, w, t)
                    # Store plot in a buffer in memory
                    img = visualize_attention.fig_to_img(fig)
                    figure = img
                    figure.savefig(os.path.join(outdir, f"{self.epoch.zfill(4)}.png"))
                    figure.clear()
            self.counter += 1


parser = argparse.ArgumentParser()
parser.add_argument('--input-source', type=str, required=True)
parser.add_argument('--bgr-source', type=str, default=None)
# only relevant if --bgr-source is specified
parser.add_argument('--model', type=str, choices=['concat', 'attention_concat', 'attention_addition'],
                    default='attention_addition')
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--load-model', type=str, required=True)
parser.add_argument('--output-type', type=str, default='png_sequence', required=False)
parser.add_argument('--resize', type=int, default=(512, 288), nargs=2)
args = parser.parse_args()

for epoch in range(len(sorted(os.listdir(args.load_model)))):
    epoch = str(epoch)
    epoch_ckpt = f"epoch-{epoch}.pth"

    # out_dir = os.path.join(args.out_dir, f"epoch-{epoch}")

    if args.model == 'attention_addition':
        visualizer = Visualizer(epoch)
        model = model_attention_addition.MattingNetwork("mobilenetv3",
                                                        pretrained_on_rvm=False,
                                                        attention_visualizer=visualizer).eval().cuda()
    elif args.model == 'attention_concat':
        visualizer = Visualizer(epoch)
        model = model_attention_concat.MattingNetwork("mobilenetv3",
                                                      pretrained_on_rvm=False,
                                                      attention_visualizer=visualizer).eval().cuda()

    model.load_state_dict(torch.load(os.path.join(args.load_model, epoch_ckpt)))
    matcher = FixedOffsetMatcher(0)

    if args.output_type == 'video':
        convert_video(
            model,  # The loaded model, can be on any device (cpu or cuda).
            input_source=args.input_source,
            input_resize=args.resize,  # [Optional] Resize the input (also the output).
            downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
            output_type='video',  # Choose "video" or "png_sequence"
            # output_composition=f"{out_dir}/com.mp4",
            # # File path if video; directory path if png sequence.
            # output_alpha=f"{out_dir}/pha.mp4",  # [Optional] Output the raw alpha prediction.
            # output_foreground=f"{out_dir}/fgr.mp4",
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
            frameids=frameids,
            epoch=int(epoch),
            matcher=matcher,
            bgr_source=args.bgr_source,
            input_resize=args.resize,  # [Optional] Resize the input (also the output).
            downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
            output_type="png_sequence",  # Choose "video" or "png_sequence"
            # output_composition=f"{out_dir}/com",
            output_alpha=os.path.join(args.out_dir, "pha"),  # [Optional] Output the raw alpha prediction.
            # bgr_src_pairs=f"{out_dir}/bgr_src",
            # output_foreground=f"{out_dir}/fgr",
            # [Optional] Output the raw foreground prediction.
            seq_chunk=12,  # Process n frames at once for better parallelism.
            progress=True  # Print conversion progress.
        )
