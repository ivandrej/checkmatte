import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from train_config import BGR_FRAME_DATA_PATHS
from .smaller_mobilenetv3 import MobileNetV3LargeEncoder
from .model_attention_addition import SpatialAttention
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import RecurrentDecoder, Projection
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner


class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_on_rvm=True,
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']

        if variant == 'mobilenetv3':
            print("Variant is mobilenetv3")
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.backbone_bgr = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(480, 128)
            # self.aspp_bgr = LRASPP(960, 128)

            # TODO: Add variables for number of channels
            self.spatial_attention = SpatialAttention(480, 480)

            self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])

        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()

        rvm_state_dict = torch.load(BGR_FRAME_DATA_PATHS["rvm_model"])
        if pretrained_on_rvm:
            print("Loading pre-trained weights from RVM")
            self.load_state_dict(rvm_state_dict, strict=False)

    def forward(self,
                src: Tensor,
                bgr: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False,
                return_intermediate=False):

        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
            bgr_sm = self._interpolate(bgr, scale_factor=downsample_ratio)
        else:
            src_sm = src
            bgr_sm = bgr

        f1, f2, f3, f4 = self.backbone(src_sm)
        f1_bgr, f2_bgr, f3_bgr, f4_bgr = self.backbone_bgr(bgr_sm)

        bgr_guidance, attention = self.spatial_attention(f4, f4_bgr, return_intermediate)
        f4_combined = bgr_guidance + f4
        f4_combined = self.aspp(f4_combined)

        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4_combined, r1, r2, r3, r4)

        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            return [fgr, pha, attention, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                              mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                              mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x

    def bgr_backbone_grads(self):
        return self.backbone_bgr.features[6].block[0][0].weight.grad

    def backbone_grads(self):
        return self.backbone.features[6].block[0][0].weight.grad

    def attention_module(self):
        return self.spatial_attention

