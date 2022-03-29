"""
"""
from typing import Optional

import torch
from torch import multiprocessing as mp, nn
from torch.cuda.amp import autocast
from torch.nn import functional as F

import sys
sys.path.append('..')

from evaluation.evaluation_metrics import MetricMAD
from visualization.visualize_attention import calc_avg_dha

from model.decoder import Projection, RecurrentDecoder
from model.deep_guided_filter import DeepGuidedFilterRefiner
from model.fast_guided_filter import FastGuidedFilterRefiner
from model.lraspp import LRASPP
from model.mobilenetv3 import MobileNetV3LargeEncoder
from model.resnet import ResNet50Encoder
from train_attention import AttentionAdditionTrainer
from train_config import BGR_FRAME_DATA_PATHS
from train_loss import pha_loss


class DebugNanGradsTrainer(AttentionAdditionTrainer):
    def init_network(self):
        self.model = MattingNetwork(self.args.model_variant,
                                    pretrained_backbone=True,
                                    pretrained_on_rvm=self.args.pretrained_on_rvm).to(
            self.rank)

        self.param_lrs = [{'params': self.model.backbone_bgr.parameters(), 'lr': self.args.learning_rate_backbone},
                          {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
                          {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
                          {'params': self.model.spatial_attention.parameters(),
                           'lr': self.args.learning_rate_backbone},
                          {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
                          {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp}]

    def train_mat(self, true_fgr, true_pha, true_bgr, precaptured_bgr, downsample_ratio, tag):
        true_fgr = true_fgr.to(self.rank, non_blocking=True)
        true_pha = true_pha.to(self.rank, non_blocking=True)
        true_bgr = true_bgr.to(self.rank, non_blocking=True)
        # Uncomment for random cropping of composited images
        # true_fgr, true_pha, true_bgr = self.random_crop(true_fgr, true_pha, true_bgr)
        true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)

        with autocast(enabled=not self.args.disable_mixed_precision):
            _, pred_pha, attention = self.model_ddp(true_src, precaptured_bgr, downsample_ratio=downsample_ratio)[:3]
            loss = pha_loss(pred_pha, true_pha)

        try:
            self.scaler.scale(loss['total']).backward()

            for name, param in self.model_ddp.module.backbone_bgr.named_parameters():
                if param.requires_grad:
                    grad_norm = torch.linalg.vector_norm(torch.flatten(param.grad.detach()))
                    weight_norm = torch.linalg.vector_norm(torch.flatten(param.detach()))
                    print(f"{name} grad norm: {grad_norm}")
                    print(f"{name} weight norm: {weight_norm}")
        except Exception as e:
            self.save()  # Save model exactly when it fails
            for name, param in self.model_ddp.module.backbone_bgr.named_parameters():
                if param.requires_grad:
                    grad_norm = torch.linalg.vector_norm(torch.flatten(param.grad.detach()))
                    weight_norm = torch.linalg.vector_norm(torch.flatten(param.detach()))
                    print(f"{name} grad norm: {grad_norm}")
                    print(f"{name} weight norm: {weight_norm}")
            # self.log_train_predictions(precaptured_bgr, pred_pha, true_pha, true_src)
            # self.attention_visualizer(attention[0], true_src[0].detach().cpu(), precaptured_bgr[0].detach().cpu(),
            #                           self.step, 'train')
            # train_avg_dha = calc_avg_dha(attention)
            # self.writer.add_scalar(f'train_{tag}_attention_dha', train_avg_dha, self.step)
            # self.test_on_random_bgr(true_src, true_pha, pred_pha, downsample_ratio=1, tag='train')
            raise e

        self.log_grad_norms()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.rank == 0 and self.step % self.args.log_train_loss_interval == 0:
            for loss_name, loss_value in loss.items():
                self.writer.add_scalar(f'train_{tag}_{loss_name}', loss_value, self.step)

            train_mad = MetricMAD()(pred_pha, true_pha)
            self.writer.add_scalar(f'train_{tag}_pha_mad', train_mad, self.step)

        if self.rank == 0 and self.step % self.args.log_train_images_interval == 0:
            self.log_train_predictions(precaptured_bgr, pred_pha, true_pha, true_src)
            # self.attention_visualizer(attention[0], true_src[0].detach().cpu(), precaptured_bgr[0].detach().cpu(),
            #                           self.step, 'train')
            # train_avg_dha = calc_avg_dha(attention)
            # self.writer.add_scalar(f'train_{tag}_attention_dha', train_avg_dha, self.step)
            # self.test_on_random_bgr(true_src, true_pha, pred_pha, downsample_ratio=1, tag='train')


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(
        DebugNanGradsTrainer,
        nprocs=world_size,
        args=(world_size,),
        join=True)


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
            self.aspp = LRASPP(960, 128)
            # self.aspp_bgr = LRASPP(960, 128)

            # TODO: Add variables for number of channels
            self.spatial_attention = SpatialAttention(960, 960)

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
                src: torch.Tensor,
                bgr: torch.Tensor,
                r1: Optional[torch.Tensor] = None,
                r2: Optional[torch.Tensor] = None,
                r3: Optional[torch.Tensor] = None,
                r4: Optional[torch.Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):

        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
            bgr_sm = self._interpolate(bgr, scale_factor=downsample_ratio)
        else:
            src_sm = src
            bgr_sm = bgr

        f1, f2, f3, f4 = self.backbone(src_sm)
        # print("F1 range: ", torch.min(f1), torch.max(f1))
        # print("F2 range: ", torch.min(f2), torch.max(f2))
        # print("F3 range: ", torch.min(f3), torch.max(f3))
        # print("F4 range: ", torch.min(f4), torch.max(f4))

        f1_bgr, f2_bgr, f3_bgr, f4_bgr = self.backbone_bgr(bgr_sm)

        # print("F1 bgr range: ", torch.min(f1_bgr), torch.max(f1_bgr))
        # print("F2 bgr range: ", torch.min(f2_bgr), torch.max(f2_bgr))
        # print("F3 bgr range: ", torch.min(f3_bgr), torch.max(f3_bgr))
        # print("F4 bgr range: ", torch.min(f4_bgr), torch.max(f4_bgr))

        bgr_guidance, attention = self.spatial_attention(f4, f4_bgr)
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

    def _interpolate(self, x: torch.Tensor, scale_factor: float):
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


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward_single_frame(self, p, b):
        assert (p.shape == b.shape)
        H, W = p.shape[-2:]
        # query = person frames, key = background frames, value = background frames

        query = self.query_conv(p).flatten(2, 3).to(dtype=torch.float32)  # B x C x N
        query = query.permute(0, 2, 1)  # B x N x C
        key = self.key_conv(b).flatten(2, 3).to(dtype=torch.float32)  # B x C x N

        with autocast(enabled=False):

            energy = torch.bmm(query, key)  # B x N x N
            attention = self.softmax(energy)  # B x N x N

        value = b.flatten(2, 3).permute(0, 2, 1)  # B x C x N --> B x N x C
        out = torch.bmm(attention, value)  # B x N x C
        out = out.permute(0, 2, 1)  # B x C x N
        out = out.unflatten(-1, (H, W))  # B x C x H x W

        return out, attention

    def forward_time_series(self, p, b):
        assert (p.shape == b.shape)
        B, T, _, H, W = p.shape
        features, attention = self.forward_single_frame(p.flatten(0, 1), b.flatten(0, 1))
        features = features.unflatten(0, (B, T))
        # attention = attention.unflatten(0, (B, T))
        attention = attention.detach().cpu().view(B, T, H, W, H, W).numpy()
        return features, attention

    def forward(self, p, b):
        assert (p.shape == b.shape)
        if p.ndim == 5:
            return self.forward_time_series(p, b)
        else:
            return self.forward_single_frame(p, b)
