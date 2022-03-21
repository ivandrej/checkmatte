import os
import sys
sys.path.append('..')
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from inference_utils import ImageSequenceReader

from inference import FixedOffsetMatcher, auto_downsample_ratio

import argparse

from evaluation.perform_experiment import get_model
from visualization.visualize_attention_all_maximum_activations import AllMaximumActivationsVisualizer

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

# Params for attention rectangle
parser.add_argument('--frameidx', type=int, required=True)
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

def convert_video(model,
                  input_source: str,
                  matcher: FixedOffsetMatcher = None,
                  bgr_source: str = None,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  bgr_src_pairs: Optional[str] = None,
                  output_attention: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        bgr_source: If provided, use model with additional background input
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """

    assert downsample_ratio is None or (
                downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    # assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'

    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if os.path.isfile(input_source):
        raise Exception("Video not implemented yet")
        # source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)

    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)
    # Read all bgr frames in memory
    bgrs = []
    for frame in sorted(os.listdir(bgr_source)):
        with Image.open(os.path.join(bgr_source, frame)) as bgr:
            bgr.load()
        if transform is not None:
            bgr = transform(bgr)
        bgrs.append(bgr)

    if output_attention is not None:
        attention_visualizer = AllMaximumActivationsVisualizer(output_attention, args.frameidx)

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device

    with torch.no_grad():
        rec = [None] * 4

        i = 0
        for src in reader:
            bgr = []
            for _ in src:  # For each of the T frames
                matched_i = min(matcher.match(i), len(bgrs) - 1)
                bgr.append(bgrs[matched_i])
                i += 1
            bgr = torch.stack(bgr)  # List of T [C, H, W] to tensor of [T, C, H, W]

            if downsample_ratio is None:
                downsample_ratio = auto_downsample_ratio(*src.shape[2:])

            src = src.to(device, dtype, non_blocking=True).unsqueeze(0)  # [B, T, C, H, W]
            bgr = bgr.to(device, dtype, non_blocking=True).unsqueeze(0)  # [B, T, C, H, W]
            fgr, pha, attention, *rec = model(src, bgr, *rec, downsample_ratio)
            if output_attention is not None:
                attention_visualizer(attention[0], src[0], bgr[0])

for epoch in args.epochs:
    out_dir = os.path.join(args.out_dir, f"epoch-{epoch}")

    model = get_model(args.model_type).eval().cuda()
    model.load_state_dict(torch.load(os.path.join(args.load_model, f"epoch-{epoch}.pth")))
    matcher = FixedOffsetMatcher(args.temporal_offset)

    convert_video(
        model,  # The loaded model, can be on any device (cpu or cuda).
        input_source=args.input_source,
        matcher=matcher,
        bgr_source=args.bgr_source,
        input_resize=args.resize,  # [Optional] Resize the input (also the output).
        downsample_ratio=None,  # [Optional] If None, make downsampled max size be 512px.
        output_type="png_sequence",  # Choose "video" or "png_sequence"
        # output_composition=f"{out_dir}/com",
        # output_alpha=f"{out_dir}/pha",  # [Optional] Output the raw alpha prediction.
        # bgr_src_pairs=f"{out_dir}/bgr_src",
        output_attention=f"{out_dir}/all_maximum_attention_activations",
        # output_foreground=f"{out_dir}/fgr",
        # [Optional] Output the raw foreground prediction.
        seq_chunk=12,  # Process n frames at once for better parallelism.
        progress=True  # Print conversion progress.
    )
