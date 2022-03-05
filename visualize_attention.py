import io
import os

import numpy
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from matplotlib.patches import Rectangle
from torchvision.utils import make_grid

"""
  /outdir
    /attention_visualization
      - /0-0 (person h = 0, w = 0)
         0.png (frame = 0) 
         .
         .
         .
"""


class Visualizer:
    def __init__(self, outdir):
        self.outdir = outdir
        self.frameidx = 0

    # First frame of sequence only
    # All pixels
    def __call__(self, attention):
        assert (attention.dim() == 5)
        T, H, W, _, _ = attention.shape

        # h = H // 2
        # w = W // 2
        for h in range(0, H, 3):
            for w in range(0, W, 3):
                figure = plot_attention(attention, h, w, 0)

                frame_out_dir = os.path.join(self.outdir, "attention_visualization", f"{h}-{w}")
                if not os.path.exists(frame_out_dir):
                    os.makedirs(frame_out_dir)

                figure.savefig(os.path.join(frame_out_dir, f"{self.frameidx}.png"))
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
    attention_matrix = attention[t][h][w]
    attention_matrix = attention_matrix * 100

    ax = sns.heatmap(attention_matrix, linewidth=0.5, linecolor='green', annot=True, fmt=".1f")

    # h and w are swapped in pyplot plots compared to numpy arrays
    ax.add_patch(Rectangle((w, h), 1, 1, fill=False, edgecolor='white', linewidth=2))
    figure = ax.get_figure()

    # Store plot in a buffer in memory
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)

    # Read plot as np img from buffer
    img = np.array(Image.open(buf)).astype(np.uint8)
    img = img[:, :, :3]  # remove alpha channel

    buf.close()
    figure.clear()
    return img


"""
    DHA = distance of highest activation
"""
def metric_DHA(attention_matrix, h, w):
    max_h, max_w = matrix_argmax(attention_matrix)

    return min(np.abs(h - max_h), np.abs(w - max_w))


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
