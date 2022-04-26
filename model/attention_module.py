import torch
from torch import nn
from torch.cuda.amp import autocast


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward_single_frame(self, p, b):
        assert(p.shape == b.shape)
        H, W = p.shape[-2:]
        # query = person frames, key = background frames, value = background frames
        query = self.query_conv(p).flatten(2, 3).to(dtype=torch.float32)  # B x C x N
        query = query.permute(0, 2, 1)  # B x N x C
        key = self.key_conv(b).flatten(2, 3).to(dtype=torch.float32)  # B x C x N

        # Disabling autocast makes sure the operations are performed with float32 precision.
        # The values of energy will overflow float16
        with autocast(enabled=False):
            energy = torch.bmm(query, key)  # B x N x N
            attention = self.softmax(energy)  # B x N x N

        value = b.flatten(2, 3).permute(0, 2, 1)  # B x C x N --> B x N x C
        out = torch.bmm(attention, value)  # B x N x C
        out = out.permute(0, 2, 1)  # B x C x N
        out = out.unflatten(-1, (H, W))  # B x C x H x W

        intermediate = {
            'attention': attention,
            'energy': energy
        }
        return out, intermediate

    """
        If return_intermediate = True, returns attention and energy as a dict
        If return_intermediate = False, returns empty dict
    """
    def forward_time_series(self, p, b, return_intermediate):
        assert (p.shape == b.shape)
        B, T, _, H, W = p.shape
        features, intermediate = self.forward_single_frame(p.flatten(0, 1), b.flatten(0, 1))
        features = features.unflatten(0, (B, T))

        # intermediate outputs are used only for visualization and debugging
        if return_intermediate:
            intermediate['attention'] = intermediate['attention'].detach().cpu().view(B, T, H, W, H, W).numpy()
            intermediate['energy'] = intermediate['energy'].detach().cpu().view(B, T, H, W, H, W).numpy()
            return features, intermediate
        else:
            return features, {}

    def forward(self, p, b, return_intermediate):
        assert(p.shape == b.shape)
        if p.ndim == 5:
            return self.forward_time_series(p, b, return_intermediate)
        else:
            return self.forward_single_frame(p, b)