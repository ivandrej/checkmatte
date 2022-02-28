"""
"""

import torch
from torch import multiprocessing as mp

from base_attention_trainer import AbstractAttentionTrainer
from model import model_attention_addition, model_attention_concat


class AttentionAdditionTrainer(AbstractAttentionTrainer):
    def init_network(self):
        if self.args.model_type == 'addition':
            self.model = model_attention_addition.MattingNetwork(self.args.model_variant,
                                                                 pretrained_backbone=True,
                                                                 pretrained_on_rvm=self.args.pretrained_on_rvm).to(
                self.rank)

            self.param_lrs = [{'params': self.model.backbone_bgr.parameters(), 'lr': self.args.learning_rate_backbone},
                              {'params': self.model.aspp_bgr.parameters(), 'lr': self.args.learning_rate_aspp},
                              # {'params': self.model.project_concat.parameters(), 'lr': self.args.learning_rate_aspp},
                              {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
                              {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
                              {'params': self.model.spatial_attention.parameters(),
                               'lr': self.args.learning_rate_backbone},
                              {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
                              {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp}]
        else:
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

    def custom_args(self, parser):
        parser.add_argument('--model-type', type=str, choices=['addition', 'concat'], default='addition')


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(
        AttentionAdditionTrainer,
        nprocs=world_size,
        args=(world_size,),
        join=True)
