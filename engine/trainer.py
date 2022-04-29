import logging
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from evaluation.metrics import dice, jaccard

class LitModel(pl.LightningModule):
    def __init__(self, model, loss, optim, post_process):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optim = optim
        self.post_process = post_process

    def forward(self, x):
        mask = self.model(x)
        return mask

    def training_step(self, batch, batch_idx):
        data, targets, indices, ratios = batch 
        preds, locations = self.model(data)
        losses = self.loss(locations, preds, targets)
        cls_loss, box_loss, center_loss = losses
        self.log('train_all_loss', sum(losses).item(), on_epoch=True)
        self.log('train_cls_loss', cls_loss.item(), on_epoch=True)
        self.log('train_box_loss', box_loss.item(), on_epoch=True)
        self.log('train_center_loss', center_loss.item(), on_epoch=True)
        return cls_loss + box_loss + center_loss

    def validation_step(self, batch, batch_idx):
        data, targets, indices, ratios = batch 
        preds, locations = self.model(data)
        losses = self.loss(locations, preds, targets)
        cls_loss, box_loss, center_loss = losses
        self.log('valid_all_loss', sum(losses).item(), on_epoch=True)
        self.log('valid_cls_loss', cls_loss.item(), on_epoch=True)
        self.log('valid_box_loss', box_loss.item(), on_epoch=True)
        self.log('valid_center_loss', center_loss.item(), on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, targets, indices, ratios = batch 
        preds, locations = self.model(data)
        losses = self.loss(locations, preds, targets)
        cls_loss, box_loss, center_loss = losses
        self.log('test_all_loss', sum(losses).item(), on_epoch=True)
        self.log('test_cls_loss', cls_loss.item(), on_epoch=True)
        self.log('test_box_loss', box_loss.item(), on_epoch=True)
        self.log('test_center_loss', center_loss.item(), on_epoch=True)

    def configure_optimizers(self):
        return self.optim


def do_train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        post_process, 
        cfg, 
    ):

    my_model = LitModel(model, loss_fn, optimizer, post_process)
    
    # ------------
    # training
    # ------------
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=cfg.SOLVER.MAX_EPOCHS)
    trainer.fit(my_model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=val_loader)
    print(result)

def do_test(
        model,
        val_loader,
        optimizer,
        loss_fn,
        post_process, 
    ):

    my_model = LitModel(model, loss_fn, optimizer, post_process)
    trainer = pl.Trainer(devices=1, accelerator="gpu")
    # ------------
    # testing
    # ------------
    result = trainer.test(model=my_model, test_dataloaders=val_loader)
    print(result)