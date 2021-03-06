"""
    Example usage:
        To train a resolution 227 x 128 F3 model:
        python train_attention.py --model-variant mobilenetv3 --dataset videomatte --resolution-lr 128 --seq-length-lr 15
        --learning-rate-backbone 0.0001 --learning-rate-aspp 0.0002 --learning-rate-decoder 0.0002 --learning-rate-refiner 0
        --checkpoint-dir  checkpoint/stage1  --log-dir log/stage1  --epoch-start 0  --epoch-end 40
        --log-train-images-interval 1000  --checkpoint-save-interval 2000 --batch-size-per-gpu 4 --temporal_offset 10
        --model-type f3 --num-workers 8

        To train a resolution 512 x 288 F4 (reduced) model:
        python train_attention.py --model-variant mobilenetv3reduced --dataset videomatte --resolution-lr 288 \
        --seq-length-lr 15 --learning-rate-backbone 0.0001 --learning-rate-aspp 0.0002 --learning-rate-decoder 0.0002 \
        --learning-rate-refiner 0 --checkpoint-dir checkpoint/stage1 --log-dir log/stage1 --epoch-start 0 --epoch-end 100 \
        --log-train-images-interval 3000 --checkpoint-save-interval 2000 --batch-size-per-gpu 4 --num-workers 8 \
        --temporal_offset 10 --model-type f4 --disable-mixed-precision
"""

import torch
from torch import multiprocessing as mp

from base_attention_trainer import AbstractAttentionTrainer
from model import model_attention_after_aspp, model_attention_concat, model_attention_f3, model_attention_f3_f2, \
    model_attention_f4, model_attention_f4_noaspp


class AttentionTrainer(AbstractAttentionTrainer):
    def init_network(self):
        if self.args.model_type == 'after_aspp':
            self.model = model_attention_after_aspp.MattingNetwork(self.args.model_variant,
                                                                 pretrained_backbone=True,
                                                                 pretrained_on_rvm=self.args.pretrained_on_rvm).to(
                self.rank)

            self.param_lrs = [{'params': self.model.backbone_bgr.parameters(), 'lr': self.args.learning_rate_backbone},
                              {'params': self.model.aspp_bgr.parameters(), 'lr': self.args.learning_rate_aspp},
                              {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
                              {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
                              {'params': self.model.spatial_attention.parameters(),
                               'lr': self.args.learning_rate_backbone},
                              {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
                              {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp}]
        elif self.args.model_type == 'concat':
            self.model = model_attention_concat.MattingNetwork(self.args.model_variant,
                                                               pretrained_backbone=True,
                                                               pretrained_on_rvm=self.args.pretrained_on_rvm).to(
                self.rank)

            self.param_lrs = [{'params': self.model.backbone_bgr.parameters(), 'lr': self.args.learning_rate_backbone},
                              {'params': self.model.aspp_bgr.parameters(), 'lr': self.args.learning_rate_aspp},
                              {'params': self.model.project_concat.parameters(), 'lr': self.args.learning_rate_aspp},
                              {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
                              {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
                              {'params': self.model.spatial_attention.parameters(),
                               'lr': self.args.learning_rate_backbone},
                              {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
                              {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp}]
        elif self.args.model_type == 'f3':
            self.model = model_attention_f3.MattingNetwork(self.args.model_variant,
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
        elif self.args.model_type == 'f4':
            self.model = model_attention_f4.MattingNetwork(self.args.model_variant,
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
        elif self.args.model_type == 'f4_noaspp':
            self.model = model_attention_f4_noaspp.MattingNetwork(self.args.model_variant,
                                                           pretrained_backbone=True,
                                                           pretrained_on_rvm=self.args.pretrained_on_rvm).to(
                self.rank)

            self.param_lrs = [{'params': self.model.backbone_bgr.parameters(), 'lr': self.args.learning_rate_backbone},
                              {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
                              {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
                              {'params': self.model.spatial_attention.parameters(),
                               'lr': self.args.learning_rate_backbone},
                              {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone}]
                              # {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp}]
        else:
            self.model = model_attention_f3_f2.MattingNetwork(self.args.model_variant,
                                                              pretrained_backbone=True,
                                                              pretrained_on_rvm=self.args.pretrained_on_rvm).to(
                self.rank)

            for att_module in self.model.spatial_attention.values():
                att_module.to(self.rank)

            self.param_lrs = [{'params': self.model.backbone_bgr.parameters(), 'lr': self.args.learning_rate_backbone},
                              {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
                              {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
                              {'params': self.model.spatial_attention['f3'].parameters(),
                               'lr': self.args.learning_rate_backbone},
                              {'params': self.model.spatial_attention['f2'].parameters(),
                               'lr': self.args.learning_rate_backbone},
                              {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
                              {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp}]

    def custom_args(self, parser):
        parser.add_argument('--model-type', type=str,
                            choices=['after_aspp', 'concat', 'f4', 'f4_noaspp', 'f3', 'f2_f3'])


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(
        AttentionTrainer,
        nprocs=world_size,
        args=(world_size,),
        join=True)
