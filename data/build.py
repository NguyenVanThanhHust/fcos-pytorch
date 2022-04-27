from os.path import join 

from torch.utils import data
from .datasets import CocoDataset
from .transforms import build_transforms

def build_datasets(data_folder, cfg, is_train=True):
    if is_train:
        datasets = CocoDataset(data_folder=join(data_folder, "train2017"), 
                                resize=cfg.INPUT.SIZE_TRAIN,
                                max_size=cfg.INPUT.MAX_SIZE, 
                                stride=cfg.MODEL.STRIDE,
                                annotations=join(data_folder, "annotations", "instances_train2017.json"),
                                training=is_train, 
                                )
    else:
        datasets = CocoDataset(data_folder=join(data_folder, "val2017"), 
                                resize=cfg.INPUT.SIZE_TEST,
                                max_size=cfg.INPUT.MAX_SIZE,
                                stride=cfg.MODEL.STRIDE,
                                annotations=join(data_folder, "annotations", "instances_val2017.json"),  
                                training=is_train, 
                                )

    return datasets

def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        shuffle = True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle = False
    datasets = build_datasets(cfg.INPUT.FOLDER, cfg, is_train=is_train)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = data.DataLoader(
        datasets, batch_size=batch_size, shuffle=shuffle, collate_fn=datasets.collate_fn, num_workers=num_workers
    )

    return data_loader