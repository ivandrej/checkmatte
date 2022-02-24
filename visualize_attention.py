import os
import seaborn as sns
from matplotlib.patches import Rectangle
import numpy as np
"""
  /outdir
    /attention_visualization
      - /0 (frame idx)
         4-7.png (person h = 4, w = 7) 
         .
         .
         .
"""

class Visualizer:
    def __init__(self, outdir):
        self.outdir = outdir
        self.frameidx = 0

    # First frame of sequence only
    # Center pixel only
    def __call__(self, attention):
        assert(attention.dim() == 5)
        T, H, W, _, _ = attention.shape

        h, w = H // 2, W // 2
        attention_matrix = attention[0][h][w].detach().cpu()
        attention_matrix = attention_matrix * 100
        # attention_matrix = np.round_(attention_matrix * 100, decimals=2)
        ax = sns.heatmap(attention_matrix, linewidth=0.5, linecolor='green', annot=True, fmt=".1f")
        # h and w are swapped in pyplot plots compared to numpy arrays
        ax.add_patch(Rectangle((w, h), 1, 1, fill=False, edgecolor='red', linewidth=1))
        figure = ax.get_figure()
        frame_out_dir = os.path.join(self.outdir, "attention_visualization", f"{self.frameidx}")
        if not os.path.exists(frame_out_dir):
            os.makedirs(frame_out_dir)

        figure.savefig(os.path.join(frame_out_dir, f"{h}-{w}.png"))
        figure.clear()
        self.frameidx += T
