import torch
import os

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm

from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter, ImagePairSequenceWriter


from model import model_attention_f4, model_attention_after_aspp, model_attention_concat, model_attention_f3


class FixedOffsetMatcher:
    def __init__(self, offset):
        self.offset = offset

    """
        i: index of person video
    """
    def match(self, i):
        return i + self.offset


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
        # print(bgr_source)
        # print(frame)
        with Image.open(os.path.join(bgr_source, frame)) as bgr:
            bgr.load()
        if transform is not None:
            bgr = transform(bgr)
        bgrs.append(bgr)

    # Initialize writers
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')
        if bgr_src_pairs is not None:
            writer_bgr = ImagePairSequenceWriter(bgr_src_pairs, 'png')

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device

    if (output_composition is not None) and (output_type == 'video'):
        # TODO: Re-name, name is overriden right now
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)

    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
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

                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                if output_alpha is not None:
                    writer_pha.write(pha[0])
                if output_composition is not None:
                    if output_type == 'video':
                        com = fgr * pha + bgr * (1 - pha)
                    else:
                        fgr = fgr * pha.gt(0)
                        com = torch.cat([fgr, pha], dim=-3)
                    writer_com.write(com[0])
                if bgr_src_pairs is not None:
                    writer_bgr.write(bgr[0], src[0])

                bar.update(src.size(1))
    except Exception as e:
        print("Failing at frame: ", i)
        raise e
    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()

def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


def get_model(model_type):
    if model_type == 'f4':
        model = model_attention_f4.MattingNetwork("mobilenetv3reduced",  # reduced encoder by default
                                                   pretrained_backbone=False,
                                                   pretrained_on_rvm=False)
    elif model_type == 'addition':
        model = model_attention_after_aspp.MattingNetwork("mobilenetv3",
                                                             pretrained_backbone=False,
                                                             pretrained_on_rvm=False)
    elif model_type == 'concat':
        model = model_attention_concat.MattingNetwork("mobilenetv3",
                                                             pretrained_backbone=False,
                                                             pretrained_on_rvm=False)
    else:
        model = model_attention_f3.MattingNetwork("mobilenetv3",
                                                             pretrained_backbone=False,
                                                             pretrained_on_rvm=False)

    return model