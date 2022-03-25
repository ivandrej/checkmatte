import os

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from visualization.visualize_attention import tensor_to_pyplot_np, matrix_argmax

COLORS = [
    (225, 0, 0),  # red
    (0, 128, 0),  # green
    (255, 153, 255),  # pink
    # (0, 0, 0),  # black
    (255, 255, 255),  # white
    (255, 255, 0)  # yellow
]

"""
    Line connecting each person location with the background location
    with the highest activation in the attention map.
"""
class AllMaximumActivationsVisualizer:
    def __init__(self, outdir, target_frame_idx):
        self.outdir = outdir
        self.frameidx = 0
        self.target_frameidx = target_frame_idx

    # First frame of sequence only
    # All pixels
    def __call__(self, attention, person, bgr):
        assert (attention.ndim == 5)
        T, H, W, _, _ = attention.shape

        for t in range(T):
            if self.frameidx == self.target_frameidx:
                img = np.concatenate((tensor_to_pyplot_np(person[t]),
                                      tensor_to_pyplot_np(bgr[t])), axis=1)
                img = img.copy()  # some unsolved opencv issue, have to do this
                img = cv2.resize(img, dsize=(910 * 2, 512), interpolation=cv2.INTER_CUBIC)

                # src image is a scaled up version of the feature map
                # The ratios can differ by at most one because we sometimes halve odd dimensions during downsampling
                # in the encoder
                assert (img.shape[0] // H - img.shape[1] // 2 // W <= 1)
                scale_factor = img.shape[0] // H
                colors = COLORS

                # Plot all lines in a rectangle
                # (h1, w1) - top left corner, (h2, w2) - bottom right
                print("H, W:", H, W)
                w1_perc, w2_perc = 0.35, 0.5
                h1_perc, h2_perc = 0.3, 0.45
                w1, w2, w_step = int(W * w1_perc), int(W * w2_perc), 1
                h1, h2, h_step = int(H * h1_perc), int(H * h2_perc), 1
                print(w1, w2)
                for w in range(w1, w2, w_step):
                    # color = np.random.random(size=3) * 256
                    color = colors[w % len(colors)]
                    for h in range(h1, h2, h_step):
                        h_, w_ = matrix_argmax(attention[t][h][w])
                        # print(f"{h}, {w} --> {h_}, {w_}")
                        cv2.line(img,
                                 (to_imgcoord(w, scale_factor), to_imgcoord(h, scale_factor)),
                                 (img.shape[1] // 2 + to_imgcoord(w_, scale_factor), to_imgcoord(h_, scale_factor)),
                                 color,
                                 thickness=2)
                os.makedirs(self.outdir, exist_ok=True)
                img = Image.fromarray(img)
                imgname = f"{self.target_frameidx}-[{w1_perc}, {w2_perc}][{h1_perc}, {h2_perc}].png"
                print(f"Saving to {imgname}")
                img.save(os.path.join(self.outdir,  imgname))
                plt.close()
            self.frameidx += 1

"""
    For a coordinate of a feature map spatial location (h or w), there exists a corresponding
    rectangle patch in the original image.
    Returns the coordinate (h or w) of the centre of this rectangle 
"""
def to_imgcoord(c, scale_factor):
    # c * scale_factor is the upper corner of each rectangle
    # Adding scale_factor // 2 brings us to the center of the rectangle
    return c * scale_factor + scale_factor // 2