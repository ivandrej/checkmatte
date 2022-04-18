"""
    Expected directory structure
    real_test_videos/
        cinema (clipgroup)
            andrej_cinema_jump/
                person/
                bgr/
            francesca_cinema_longwalk
                person/
                bgr/
            ..
"""
import argparse
import os
import sys

import torch

sys.path.append('..')
from model import rvm, model_attention_f3, model_attention_f3_f2, model_attention_f4, model_attention_f4_noaspp
import rvm_inference, inference

parser = argparse.ArgumentParser()
parser.add_argument('--clipgroup', type=str, required=True)
args = parser.parse_args()

input_dir = os.path.join('/media/andivanov/DATA/dynamic_backgrounds_captured/real_test_videos/', args.clipgroup)

# Reduced f4 model - res 288
model = model_attention_f4.MattingNetwork("mobilenetv3reduced", pretrained_on_rvm=False).eval().cuda()

model.load_state_dict(torch.load("/media/andivanov/DATA/euler_training/final/removefeat12to16/noscaler_float32/"
                                 "res288_lr1e4_B4_newrun/checkpoint/stage2/epoch-41.pth"))

for clip in os.listdir(input_dir):
    outdir = os.path.join("/media/andivanov/DATA/results/", args.clipgroup, clip,
                          'removefeat12to16/float32/res288_lr1e4_B4', 'epoch-41')
    matcher = inference.FixedOffsetMatcher(0)

    inference.convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=os.path.join(input_dir, clip, 'person'),
        matcher=matcher,
        bgr_source=os.path.join(input_dir, clip, 'bgr'),
        input_resize=(512, 288),  # [Optional] Resize the input (also the output).
        downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
        output_type="png_sequence",  # Choose "video" or "png_sequence"
        # output_composition=f"{out_dir}/com",
        output_alpha=f"{outdir}/pha",  # [Optional] Output the raw alpha prediction.
        bgr_src_pairs=f"{outdir}/bgr_src",
        # output_attention=f"{out_dir}/attention",
        # output_foreground=f"{out_dir}/fgr",
        # [Optional] Output the raw foreground prediction.
        seq_chunk=12,  # Process n frames at once for better parallelism.
        progress=True  # Print conversion progress.
    )

# Reduced f4 model - res 192
model = model_attention_f4.MattingNetwork("mobilenetv3reduced", pretrained_on_rvm=False).eval().cuda()

model.load_state_dict(torch.load("/media/andivanov/DATA/euler_training/final/removefeat12to16/noscaler_float32/"
                                 "res192_lr1e4_B4_newrun/checkpoint/stage1/epoch-36.pth"))

for clip in os.listdir(input_dir):
    outdir = os.path.join("/media/andivanov/DATA/results/", args.clipgroup, clip,
                          'removefeat12to16/float32/res192_lr1e4_B4', 'epoch-36')
    matcher = inference.FixedOffsetMatcher(0)

    inference.convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=os.path.join(input_dir, clip, 'person'),
        matcher=matcher,
        bgr_source=os.path.join(input_dir, clip, 'bgr'),
        input_resize=(341, 192),  # [Optional] Resize the input (also the output).
        downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
        output_type="png_sequence",  # Choose "video" or "png_sequence"
        # output_composition=f"{out_dir}/com",
        output_alpha=f"{outdir}/pha",  # [Optional] Output the raw alpha prediction.
        bgr_src_pairs=f"{outdir}/bgr_src",
        # output_attention=f"{out_dir}/attention",
        # output_foreground=f"{out_dir}/fgr",
        # [Optional] Output the raw foreground prediction.
        seq_chunk=12,  # Process n frames at once for better parallelism.
        progress=True  # Print conversion progress.
    )

model = model_attention_f4.MattingNetwork("mobilenetv3reduced", pretrained_on_rvm=False).eval().cuda()

model.load_state_dict(torch.load("/media/andivanov/DATA/euler_training/final/removefeat12to16/noscaler_float32/"
                                 "res192_lr1e4_B4_newrun/checkpoint/stage1/epoch-49.pth"))

for clip in os.listdir(input_dir):
    outdir = os.path.join("/media/andivanov/DATA/results/", args.clipgroup, clip,
                          'removefeat12to16/float32/res192_lr1e4_B4', 'epoch-49')
    matcher = inference.FixedOffsetMatcher(0)

    inference.convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=os.path.join(input_dir, clip, 'person'),
        matcher=matcher,
        bgr_source=os.path.join(input_dir, clip, 'bgr'),
        input_resize=(341, 192),  # [Optional] Resize the input (also the output).
        downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
        output_type="png_sequence",  # Choose "video" or "png_sequence"
        # output_composition=f"{out_dir}/com",
        output_alpha=f"{outdir}/pha",  # [Optional] Output the raw alpha prediction.
        bgr_src_pairs=f"{outdir}/bgr_src",
        # output_attention=f"{out_dir}/attention",
        # output_foreground=f"{out_dir}/fgr",
        # [Optional] Output the raw foreground prediction.
        seq_chunk=12,  # Process n frames at once for better parallelism.
        progress=True  # Print conversion progress.
    )

# RVM reduced res = 192
model = rvm.MattingNetwork("mobilenetv3reduced").eval().cuda()
model.load_state_dict(torch.load("/media/andivanov/DATA/euler_training/final/removefeat12to16/rvm/fromscratch/"
                                 "res192_lr1e4_B4/checkpoint/stage1/epoch-38.pth"))

for clip in os.listdir(input_dir):
    outdir = os.path.join("/media/andivanov/DATA/results/", args.clipgroup, clip,
                          "removefeat12to16/rvm/fromscratch/res192_lr1e4_B4/epoch-38")

    rvm_inference.convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=os.path.join(input_dir, clip, 'person'),
        input_resize=(341, 192),  # [Optional] Resize the input (also the output).
        downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
        output_type="png_sequence",  # Choose "video" or "png_sequence"
        # output_composition=f"{out_dir}/com",
        output_alpha=f"{outdir}/pha",  # [Optional] Output the raw alpha prediction.
        # output_foreground=f"{out_dir}/fgr",
        # [Optional] Output the raw foreground prediction.
        seq_chunk=12,  # Process n frames at once for better parallelism.
        progress=True  # Print conversion progress.
    )

# RVM reduced res = 288
model = rvm.MattingNetwork("mobilenetv3reduced").eval().cuda()
model.load_state_dict(torch.load("/media/andivanov/DATA/euler_training/final/removefeat12to16/rvm/fromscratch/"
                                 "res288_lr1e4_B4/checkpoint/stage1/epoch-37.pth"))

for clip in os.listdir(input_dir):
    outdir = os.path.join("/media/andivanov/DATA/results/", args.clipgroup, clip,
                          "removefeat12to16/rvm/fromscratch/res288_lr1e4_B4/epoch-37")

    rvm_inference.convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=os.path.join(input_dir, clip, 'person'),
        input_resize=(512, 288),  # [Optional] Resize the input (also the output).
        downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
        output_type="png_sequence",  # Choose "video" or "png_sequence"
        # output_composition=f"{out_dir}/com",
        output_alpha=f"{outdir}/pha",  # [Optional] Output the raw alpha prediction.
        # output_foreground=f"{out_dir}/fgr",
        # [Optional] Output the raw foreground prediction.
        seq_chunk=12,  # Process n frames at once for better parallelism.
        progress=True  # Print conversion progress.
    )