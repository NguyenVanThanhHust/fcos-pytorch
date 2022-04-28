import numpy as np
import torch.nn.functional as F
from torch import nn

class FCOSHead(nn.Module):
    def __init__(self, in_channels, n_class, n_conv, priors):
        super().__init__()

        cls_tower = []
        bbox_tower = []
        for i in range(n_conv):
            cls_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, in_channel))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            bbox_tower.append(nn.GroupNorm(32, in_channel))
            bbox_tower.append(nn.ReLU())

        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)

        self.cls_pred = nn.Conv2d(in_channel, n_class, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channel, 4, 3, padding=1)
        self.center_pred = nn.Conv2d(in_channel, 1, 3, padding=1)

        self.apply(init_conv_std)

        prior_bias = -math.log((1 - prior) / prior)
        nn.init.constant_(self.cls_pred.bias, prior_bias)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])
    
    def forward(self, inputs):
        logits = []
        bboxes = []
        centers = []

        for feat, scale in zip(inputs, self.scales):
            cls_out = self.cls_tower(feat)
            bbox_out = self.bbox_tower(feat)
            center_out = self.center_pred(feat)
            
