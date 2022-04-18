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

# # Fully trained RVM
# model = model.MattingNetwork("mobilenetv3").eval().cuda()
# model.load_state_dict(torch.load("/media/andivanov/DATA/training/rvm_mobilenetv3.pth"))
#
# for clip in os.listdir(input_dir):
#     outdir = os.path.join("/media/andivanov/DATA/results/", args.clipgroup, clip, 'rvm_fromscratch_epoch39')
#
#     rvm_inference.convert_video(
#         model,  # The loaded model, can be on any device (cpu or cuda).
#         input_source=os.path.join(input_dir, clip, 'person'),
#         input_resize=(227, 128),  # [Optional] Resize the input (also the output).
#         downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
#         output_type="png_sequence",  # Choose "video" or "png_sequence"
#         # output_composition=f"{out_dir}/com",
#         output_alpha=f"{outdir}/pha",  # [Optional] Output the raw alpha prediction.
#         # output_foreground=f"{out_dir}/fgr",
#         # [Optional] Output the raw foreground prediction.
#         seq_chunk=12,  # Process n frames at once for better parallelism.
#         progress=True  # Print conversion progress.
#     )

# Test RVM
model = rvm.MattingNetwork("mobilenetv3").eval().cuda()
model.load_state_dict(torch.load("/media/andivanov/DATA/euler_training/rvm/fromscratch/"
                                 "res128_lr1e4_B4/checkpoint/stage1/epoch-39.pth"))

for clip in os.listdir(input_dir):
    outdir = os.path.join("/media/andivanov/DATA/results/", args.clipgroup, clip, 'rvm_fromscratch/epoch39')

    rvm_inference.convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=os.path.join(input_dir, clip, 'person'),
        input_resize=(227, 128),  # [Optional] Resize the input (also the output).
        downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
        output_type="png_sequence",  # Choose "video" or "png_sequence"
        # output_composition=f"{out_dir}/com",
        output_alpha=f"{outdir}/pha",  # [Optional] Output the raw alpha prediction.
        # output_foreground=f"{out_dir}/fgr",
        # [Optional] Output the raw foreground prediction.
        seq_chunk=12,  # Process n frames at once for better parallelism.
        progress=True  # Print conversion progress.
    )

model = model_attention_f3.MattingNetwork("mobilenetv3").eval().cuda()
model.load_state_dict(torch.load("/media/andivanov/DATA/euler_training/attention_variants/"
                                 "f3/fromscratch/offset10/res128_lr1e4_B4/checkpoint/stage2/epoch-38.pth"))

for clip in os.listdir(input_dir):
    outdir = os.path.join("/media/andivanov/DATA/results/", args.clipgroup, clip,
                          'attention_f3_offset10_from_scratch', 'epoch-38')
    matcher = inference.FixedOffsetMatcher(0)

    inference.convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=os.path.join(input_dir, clip, 'person'),
        matcher=matcher,
        bgr_source=os.path.join(input_dir, clip, 'bgr'),
        input_resize=(227, 128),  # [Optional] Resize the input (also the output).
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

# model = model_attention_f3_f2.MattingNetwork("mobilenetv3").eval().cuda()
# if type(model.spatial_attention) == dict:
#     for att_mod in model.spatial_attention.values():
#         att_mod.cuda()
#
# model.load_state_dict(torch.load("/media/andivanov/DATA/euler_training/roughly_matched/attention_variants/"
#                                  "f2_f3/fromscratch/offset10/res128_lr1e4_B4/checkpoint/stage1/epoch-38.pth"))
#
# for clip in os.listdir(input_dir):
#     outdir = os.path.join("/media/andivanov/DATA/results/", args.clipgroup, clip, 'attention_f2_f3_offset10_res128_from_scratch')
#     matcher = inference.FixedOffsetMatcher(0)
#
#     inference.convert_video(
#         model,  # The loaded model, can be on any device (cpu or cuda).
#         input_source=os.path.join(input_dir, clip, 'person'),
#         matcher=matcher,
#         bgr_source=os.path.join(input_dir, clip, 'bgr'),
#         input_resize=(227, 128),  # [Optional] Resize the input (also the output).
#         downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
#         output_type="png_sequence",  # Choose "video" or "png_sequence"
#         # output_composition=f"{out_dir}/com",
#         output_alpha=f"{outdir}/pha",  # [Optional] Output the raw alpha prediction.
#         bgr_src_pairs=f"{outdir}/bgr_src",
#         # output_attention=f"{out_dir}/attention",
#         # output_foreground=f"{out_dir}/fgr",
#         # [Optional] Output the raw foreground prediction.
#         seq_chunk=12,  # Process n frames at once for better parallelism.
#         progress=True  # Print conversion progress.
#     )



# Reduced f4 no aspp model
# model = model_attention_f4_noaspp.MattingNetwork("mobilenetv3reduced", pretrained_on_rvm=False).eval().cuda()
#
# model.load_state_dict(torch.load("/media/andivanov/DATA/euler_training/attention_variants/f4/"
#                                  "removefeat12to16_removeaspp/float32/res192_lr1e4_B4/res192_lr1e4_B4/checkpoint/stage1/epoch-14.pth"))
#
# for clip in os.listdir(input_dir):
#     outdir = os.path.join("/media/andivanov/DATA/results/", args.clipgroup, clip,
#                           'removefeat12to16_removeaspp/float32/res192_lr1e4_B4', 'epoch-14')
#     matcher = inference.FixedOffsetMatcher(0)
#
#     inference.convert_video(
#         model,  # The loaded model, can be on any device (cpu or cuda).
#         input_source=os.path.join(input_dir, clip, 'person'),
#         matcher=matcher,
#         bgr_source=os.path.join(input_dir, clip, 'bgr'),
#         input_resize=(341, 192),  # [Optional] Resize the input (also the output).
#         downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
#         output_type="png_sequence",  # Choose "video" or "png_sequence"
#         # output_composition=f"{out_dir}/com",
#         output_alpha=f"{outdir}/pha",  # [Optional] Output the raw alpha prediction.
#         bgr_src_pairs=f"{outdir}/bgr_src",
#         # output_attention=f"{out_dir}/attention",
#         # output_foreground=f"{out_dir}/fgr",
#         # [Optional] Output the raw foreground prediction.
#         seq_chunk=12,  # Process n frames at once for better parallelism.
#         progress=True  # Print conversion progress.
#     )


# Heavy transformations res 128 model
# model = model_attention_f3.MattingNetwork("mobilenetv3").eval().cuda()
# model.load_state_dict(torch.load("/media/andivanov/DATA/euler_training/final/"
#                                  "bgronlytransform_heavy/f3/res128_lr1e4_B4/checkpoint/stage2/epoch-56.pth"))
#
# for clip in os.listdir(input_dir):
#     outdir = os.path.join("/media/andivanov/DATA/results/", args.clipgroup, clip,
#                           'bgronlytransform_heavy/f3/res128_lr1e4_B4', 'epoch-56')
#     matcher = inference.FixedOffsetMatcher(0)
#
#     inference.convert_video(
#         model,  # The loaded model, can be on any device (cpu or cuda).
#         input_source=os.path.join(input_dir, clip, 'person'),
#         matcher=matcher,
#         bgr_source=os.path.join(input_dir, clip, 'bgr'),
#         input_resize=(227, 128),  # [Optional] Resize the input (also the output).
#         downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
#         output_type="png_sequence",  # Choose "video" or "png_sequence"
#         # output_composition=f"{out_dir}/com",
#         output_alpha=f"{outdir}/pha",  # [Optional] Output the raw alpha prediction.
#         bgr_src_pairs=f"{outdir}/bgr_src",
#         # output_attention=f"{out_dir}/attention",
#         # output_foreground=f"{out_dir}/fgr",
#         # [Optional] Output the raw foreground prediction.
#         seq_chunk=12,  # Process n frames at once for better parallelism.
#         progress=True  # Print conversion progress.
#     )