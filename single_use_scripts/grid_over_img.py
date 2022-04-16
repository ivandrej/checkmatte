import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# path = "/media/andivanov/DATA/experiments/inputs/VMx3x30FPS_2x60FPS/0002_stairs/0180.png"
path = "/media/andivanov/DATA/dynamic_backgrounds_captured/img_seq/stairs/0230.png"
img = Image.open(path)
# fig = plt.figure(figsize=(float(img.size[0])/50,float(img.size[1])/50),dpi=200)
fig = plt.figure(figsize=(10.24, 5.76), dpi=250)
# print(float(img.size[0])/50,float(img.size[1])/50)
ax = fig.add_subplot(111)
H, W = 16, 29
heatmap_grid = np.zeros((16, 29))
sns.heatmap(heatmap_grid, cmap="cubehelix", linewidth=0.1, linecolor='green',
                     annot=False, fmt=".1f", zorder=4, alpha=0, cbar=False, ax=ax)

for h in range(H):
    for w in range(W):
        ax.add_patch(Rectangle((w, h), 1, 1, fill=False, edgecolor='green', linewidth=0.2, zorder=5))

# ax.add_patch(Rectangle((19, 3), 1, 1, fill=False, edgecolor='red', linewidth=1, zorder=5))

# Add the image
ax.imshow(img,
          aspect=ax.get_aspect(),
          extent=ax.get_xlim() + ax.get_ylim(),
          zorder=1)
# plt.show()
outpath = "/media/andivanov/DATA/report/single_person_location_attention_vis/chosen/" \
          "0002_stairs/3-19/bgrlocation/bgr.png"
plt.savefig(outpath, bbox_inches='tight')
