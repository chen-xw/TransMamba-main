# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

# from ..builder import HEADS
# from .cls_head import ClsHead


# @HEADS.register_module()
class LinearClsHead(nn.Module):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 bias = True,
              ):
        super(LinearClsHead, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes, bias=bias)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def forward(self, x):
        x = self.pre_logits(x)
        cls_score = self.fc(x)
        # losses = self.loss(cls_score, gt_label, **kwargs)
        return cls_score
