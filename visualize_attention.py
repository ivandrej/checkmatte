import os
import seaborn as sns


class Visualizer:
    def __init__(self, outdir):
        self.outdir = outdir
        self.frameidx = 0

    def __call__(self, attention, H, W):
        assert(attention.dim() == 3)
        T, N, _ = attention.shape
        if self.frameidx == 0:
            centeridx = N // 2
            attention_matrix = attention[0][centeridx].view(H, W).detach().cpu()

            ax = sns.heatmap(attention_matrix)
            figure = ax.get_figure()
            figure.savefig(os.path.join(self.outdir, f"frame{self.frameidx}.png"))

        self.frameidx += T
