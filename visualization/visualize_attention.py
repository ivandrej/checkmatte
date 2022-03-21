import io
import os

import numpy
import numpy as np
import seaborn as sns
import torch
from PIL import Image
import matplotlib.pyplot as plt
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
    Plots a sequence of attention to Tensorboard (not a batch of sequences):
      - 4 key locations
      - all frames
"""
class TrainVisualizer:
    def __init__(self, writer):
        self.writer = writer

    """
        attention, person, bgr: torch.Tensor
        Assumes single batch (no batch dimension)
        Assumes all tensors are on CPU, so they can be converted to numpy
    """
    def __call__(self, attention, person, bgr, step, tag):
        assert (attention.ndim == 5)
        assert (person.ndim == 4)
        assert (bgr.ndim == 4)

        T, H, W, _, _ = attention.shape

        for h, w in key_spatial_locations(H, W):
            figures = []
            for t in range(T):
                figure = sidebysidevisualize(attention, bgr, h, person, t, w)
                figures.append(figure)

            # Add batch dim, currently ignored in plot method
            figures = torch.from_numpy(np.array(figures)).unsqueeze(0)
            figures = figures.permute(0, 1, 4, 2, 3)  # B, T, H, W, C  --> B, T, C, H, W
            self.writer.add_image(f'{tag}_attention_{h},{w}',
                                  make_grid(figures.flatten(0, 1), nrow=figures.size(1)),
                                  step)

"""
    Visualizes the attention map for all person anchor locations in 
    a specified rectangle
"""
class RectangleVisualizer:

    def __init__(self, outdir, target_frame_idx, rec_h, rec_w, rec_size_h, rec_size_w):
        self.outdir = outdir
        self.frameidx = 0
        self.target_frameidx = target_frame_idx
        self.rec_w = rec_w
        self.rec_h = rec_h
        self.rec_size_w = rec_size_w
        self.rec_size_h = rec_size_h

    # First frame of sequence only
    # All pixels
    def __call__(self, attention, person, bgr):
        assert (attention.ndim == 5)
        T, H, W, _, _ = attention.shape

        for t in range(T):
            if self.frameidx == self.target_frameidx:
                for h in range(self.rec_h, self.rec_h + self.rec_size_h):
                    for w in range(self.rec_w, self.rec_w + self.rec_size_w):
                        res = sidebysidevisualize(attention, bgr, h, person, t, w)

                        frame_out_dir = os.path.join(self.outdir, f"{self.target_frameidx}")
                        os.makedirs(frame_out_dir, exist_ok=True)
                        Image.fromarray(res).save(os.path.join(frame_out_dir, f"{h}-{w}.png"))
                        plt.close()
            self.frameidx += 1

"""
    Visualizes what a particular person location focuses on over the background frame
    Plots:
      - Left: person frame with the anchor person location
      - Right: the background frame overlayed with the attention map
      
      attention, bgr, person: torch.Tensor 
"""
def sidebysidevisualize(attention, bgr, h, person, t, w):
    assert (bgr.dim() == 4)
    assert (person.dim() == 4)
    #  Bgr frame overlayed with attention map
    fig = plt.figure()
    bgr_np = tensor_to_pyplot_np(bgr[t])
    get_attention_over_bgr_fig(attention, bgr_np, h, w, t)

    heatmap_img = fig_to_img(fig)
    ax = plt.gca()
    aspect = ax.get_aspect()
    extent = ax.get_xlim() + ax.get_ylim()
    plt.close(fig)

    # Person anchor location shown on person frame
    fig = plt.figure()
    ax = plt.gca()
    person_np = tensor_to_pyplot_np(person[t])
    ax.imshow(person_np, aspect=aspect, extent=extent)
    ax.add_patch(Rectangle((w, h), 1, 1, fill=False, edgecolor='red', linewidth=1))
    person_anchor_img = fig_to_img(fig)
    plt.close(fig)
    # Show person frame left and bgr frame right
    res = np.concatenate((person_anchor_img, heatmap_img), axis=1)
    return res

"""
    Torch images during training are CHW [0,1]. Converts them to numpy HWC [0, 255] 
"""
def tensor_to_pyplot_np(x):
    x = x.permute(1, 2, 0) * 255
    x = x.cpu().detach().numpy().astype(np.uint8)
    return x

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
    # figure.clear()
    return img

def get_attention_fig(attention, h, w, t):
    attention_matrix = attention[t][h][w]
    attention_matrix = attention_matrix * 100
    ax = sns.heatmap(attention_matrix, linewidth=0.1, linecolor='green', annot=False, fmt=".1f")
    # h and w are swapped in pyplot plots compared to numpy arrays
    ax.add_patch(Rectangle((w, h), 1, 1, fill=False, edgecolor='white', linewidth=2))
    figure = ax.get_figure()
    return figure

def get_attention_over_bgr_fig(attention, bgr, h, w, t):
    attention_matrix = attention[t][h][w]
    attention_matrix = attention_matrix * 100
    ax = sns.heatmap(attention_matrix, cmap="Blues", linewidth=0.1, linecolor='green',
                     annot=False, fmt=".1f", zorder=4, alpha=0.6)
    # h and w are swapped in pyplot plots compared to numpy arrays
    ax.add_patch(Rectangle((w, h), 1, 1, fill=False, edgecolor='red', linewidth=1, zorder=5))
    # plt.show()
    ax.imshow(bgr,
              aspect=ax.get_aspect(),
              extent=ax.get_xlim() + ax.get_ylim(),
              zorder=1)


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
