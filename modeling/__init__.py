from .fcos import FCOS
from .loss import FCOSLoss
from .postprocess import FCOSPostprocessor

def build_model(cfg):
    model = FCOS(backbones=cfg.MODEL.BACKBONE,
                classes=cfg.MODEL.NUM_CLASSES, 
                cfg=cfg)
    return model

def build_loss(cfg):
    return FCOSLoss(cfg.SIZES, cfg.GAMMA, cfg.ALPHA, 
                cfg.CENTER_SAMPLING, cfg.FPN_STRIDES, 
                cfg.POS_RADIUS)

def build_post_process(cfg):
    return FCOSPostprocessor()