import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

import layers as backbones_mod
from modeling.fcos_head import FCOSHead

class FCOS(nn.Module):
    def __init__(self, 
        backbones='ResNet50FPN', 
        classes=80, 
        cfg=None, 
        ):
        super().__init__()
        if not isinstance(backbones, list):
            backbones = [backbones]
        self.backbones = nn.ModuleDict({b: getattr(backbones_mod, b)() for b in backbones})
        self.head = FCOSHead(
                    in_channel=cfg.MODEL.FEATURE_DEPTH, 
                    n_class=cfg.MODEL.NUM_CLASSES, 
                    n_conv=cfg.MODEL.NUM_CONV_BLOCK, 
                    prior=cfg.PRIOR, 
                    )
        self.fpn_strides = cfg.FPN_STRIDES

    def forward(self, x):
        # Backbones forward pass
        features = []
        for _, backbone in self.backbones.items():
            features.extend(backbone(x))
        pred_logits, pred_bboxes, pred_centers = self.head(features)
        locations = self.compute_location(features)
        preds = pred_logits, pred_bboxes, pred_centers
        return preds, locations

    def compute_location(self, features):
        locations = []
        for i, feat in enumerate(features):
            _, _, height, width = feat.shape
            location_per_level = self.compute_location_per_level(
                height, 
                width, 
                self.fpn_strides[i]
            )
            locations.append(location_per_level.to(features[0].device))
        return locations

    def compute_location_per_level(self, height, width, stride, ):
        shift_x = torch.arange(0, width*stride, step=stride)
        shift_y = torch.arange(0, height*stride, step=stride)
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y)

        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_x, shift_y), 1) + stride // 2
        return location