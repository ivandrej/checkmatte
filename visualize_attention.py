import os

import seaborn as sns
from matplotlib.patches import Rectangle

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
      - frame = 2 
      - first sequence in batch
"""
class TrainVisualizer:
    def __init__(self, writer):
        self.writer = writer

    def __call__(self, attention, step, tag):
        assert (attention.ndim == 6)
        attention = attention[0]
        T, H, W, _, _ = attention.shape

        frameidx = 2
        for h, w in key_spatial_locations(H, W):
            figure = plot_attention(attention, h, w, frameidx)

            self.writer.add_figure(f'attention_{tag}_{h}:{w}_frame{frameidx}', figure, step)


def plot_attention(attention, h, w, t):
    attention_matrix = attention[t][h][w]
    attention_matrix = attention_matrix * 100
    # print("Attention matrix sum: ", attention_matrix.flatten().sum())
    # assert (attention_matrix.flatten().sum().eq(100))
    # attention_matrix = np.round_(attention_matrix * 100, decimals=2)
    ax = sns.heatmap(attention_matrix, linewidth=0.5, linecolor='green', annot=True, fmt=".1f")
    # h and w are swapped in pyplot plots compared to numpy arrays
    ax.add_patch(Rectangle((w, h), 1, 1, fill=False, edgecolor='red', linewidth=1))
    figure = ax.get_figure()
    return figure


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
