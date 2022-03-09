import io
import os

import numpy
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from matplotlib.patches import Rectangle
from torchvision.utils import make_grid
import cv2
from torchvision.transforms import functional as F

"""
  /outdir
    /attention_visualization
      - /0-0 (person h = 0, w = 0)
         0.png (frame = 0) 
         .
         .
         .
"""


class TestVisualizer:
    def __init__(self, outdir):
        self.outdir = outdir
        self.frameidx = 0

    # First frame of sequence only
    # All pixels
    def __call__(self, attention):
        assert (attention.ndim == 5)
        T, H, W, _, _ = attention.shape

        for h, w in key_spatial_locations(H, W):
            for t in range(T):
                figure = get_attention_fig(attention, h, w, t)

                location_out_dir = os.path.join(self.outdir, f"{h}-{w}")
                os.makedirs(location_out_dir, exist_ok=True)

                figure.savefig(os.path.join(location_out_dir, f"{self.frameidx + t}.png"))
                figure.clear()
        self.frameidx += T

"""
    Plots to Tensorboard:
      - 4 key locations
      - first sequence in batch
      - all frames
      
    Returns avg dha score over the whole batch 
"""
class TrainVisualizer:
    def __init__(self, writer):
        self.writer = writer

    def __call__(self, attention, step, tag):
        assert (attention.ndim == 6)
        # Only take first sequence in the batch
        attention = attention[0]
        T, H, W, _, _ = attention.shape

        for h, w in key_spatial_locations(H, W):
            figures = []
            for t in range(T):
                figure = plot_attention(attention, h, w, t)
                figures.append(figure)

            # Add batch dim, currently ignored in plot method
            figures = torch.from_numpy(np.array(figures)).unsqueeze(0)
            figures = figures.permute(0, 1, 4, 2, 3)  # B, T, H, W, C  --> B, T, C, H, W
            self.writer.add_image(f'{tag}_attention_{h},{w}',
                                  make_grid(figures.flatten(0, 1), nrow=figures.size(1)),
                                  step)

class RectangleVisualizer:
    def __init__(self, outdir):
        self.outdir = outdir
        self.frameidx = 0

    # First frame of sequence only
    # All pixels
    def __call__(self, attention, person):
        assert (attention.ndim == 5)
        T, H, W, _, _ = attention.shape

        rec_h, rec_w = 4, 10
        rec_size_h, rec_size_w = 4, 10
        target_frameidx = 106
        for t in range(T):
            if self.frameidx == target_frameidx:
                for h in range(rec_h, rec_h + rec_size_h):
                    for w in range(rec_w, rec_w + rec_size_w):
                        heatmap_img = plot_attention(attention, h, w, t)
                        # Convert CHW, [0,1] --> HWC, [0, 255]
                        resized_person = F.resize(person[t], heatmap_img.shape[:2]).permute(1, 2, 0) * 255
                        resized_person = resized_person.cpu().detach().numpy().astype(np.uint8)

                        res = cv2.addWeighted(resized_person, 0.6, heatmap_img, 0.4, 0)
                        frame_out_dir = os.path.join(self.outdir, f"{target_frameidx}")
                        os.makedirs(frame_out_dir, exist_ok=True)
                        Image.fromarray(res).save(os.path.join(frame_out_dir, f"{h}-{w}.png"))
            self.frameidx += 1

"""
    DHA average over all frames of all batches.
    4 person anchor positions sampled.
"""
def calc_avg_dha(attention):
    assert (attention.ndim == 6)
    B, T, H, W, _, _ = attention.shape

    dha_total, cnt = 0, 0
    for b in range(B):
        for h, w in key_spatial_locations(H, W):
            for t in range(T):
                attention_matrix = attention[b][t][h][w]
                # print("Single DHA: ", metric_DHA(attention_matrix, h, w))
                dha_total += metric_DHA(attention_matrix, h, w)
                cnt += 1

    dha_avg = dha_total / cnt
    return dha_avg

def plot_attention(attention, h, w, t):
    figure = get_attention_fig(attention, h, w, t)

    # Store plot in a buffer in memory
    img = fig_to_img(figure)
    return img

"""
    Converts a pyplot figure to an RGB np image
"""
def fig_to_img(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    # Read plot as np img from buffer
    img = np.array(Image.open(buf)).astype(np.uint8)
    img = img[:, :, :3]  # remove alpha channel
    buf.close()
    figure.clear()
    return img

def get_attention_fig(attention, h, w, t):
    attention_matrix = attention[t][h][w]
    attention_matrix = attention_matrix * 100
    ax = sns.heatmap(attention_matrix, linewidth=0.1, linecolor='green', annot=False, fmt=".1f")
    # h and w are swapped in pyplot plots compared to numpy arrays
    ax.add_patch(Rectangle((w, h), 1, 1, fill=False, edgecolor='white', linewidth=2))
    figure = ax.get_figure()
    return figure


"""
    DHA = distance of highest activation
"""
def metric_DHA(attention_matrix, h, w):
    max_h, max_w = matrix_argmax(attention_matrix)

    return numpy.linalg.norm([max_h - h, max_w - w])  # Euclidean distance
    # return min(np.abs(h - max_h), np.abs(w - max_w))


def matrix_argmax(m: numpy.ndarray):
    assert (m.ndim == 2)
    return np.unravel_index(np.argmax(m, axis=None), m.shape)

"""
 The most important spatial locations in the person feature map.
 Used to visualize attention
"""


def key_spatial_locations(H, W):
    return [
        (H // 2, W // 2),  # center
        (3 * H // 4, W // 2),  # around feet
        (3 * H // 4, W // 4),  # bottom left corner
        (3 * H // 4, 3 * W // 4)  # bottom right corner
    ]
