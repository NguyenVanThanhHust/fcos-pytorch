import torch
from torch import nn

# from boxlist import BoxList, boxlist_nms, remove_small_box, cat_boxlist

class FCOSPostprocessor(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward_single_feature_map(self, location, pred_logit, pred_bbox, 
        pred_center, image_size):
        return 
    
    def forward(self, locations, pred_logits, pred_bboxes, 
        pred_centers, image_sizes):
        return 

