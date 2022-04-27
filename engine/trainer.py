import logging
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from evaluation.metrics import dice, jaccard

class LitModel(pl.LightningModule):
    def __init__(self, model, loss, optim):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optim = optim
        self.train_dice_scores = []
        self.train_jaccard_scores = []
        self.val_dice_scores = []
        self.val_jaccard_scores = []
        self.test_dice_scores = []
        self.test_jaccard_scores = []

    def forward(self, x):
        mask = self.model(x)
        return mask

    def training_step(self, batch, batch_idx):
        data, target = batch 
        pred = self.model(data)
        loss = self.loss(pred, target)
        dice_value = dice(pred, target)
        jaccard_value = jaccard(pred, target)
        self.train_dice_scores.append(dice_value)
        self.train_jaccard_scores.append(jaccard_value)
        self.log('train_loss', loss, on_epoch=True)
        self.log('dice_value', dice_value, on_epoch=True)
        self.log('jaccard_value', jaccard_value, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch 
        pred = self.model(data)
        loss = self.loss(pred, target)
        dice_value = dice(pred, target)
        jaccard_value = jaccard(pred, target)
        self.val_dice_scores.append(dice_value)
        self.val_jaccard_scores.append(jaccard_value)
        self.log('valid_loss', loss, on_epoch=True)
        self.log('dice_value', dice_value, on_epoch=True)
        self.log('jaccard_value', jaccard_value, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, target = batch 
        pred = self.model(data)
        loss = self.loss(pred, target)
        dice_value = dice(pred, target)
        jaccard_value = jaccard(pred, target)
        self.test_dice_scores.append(dice_value)
        self.test_jaccard_scores.append(jaccard_value)
        self.log('test_loss', loss, on_epoch=True)
        self.log('dice_value', dice_value, on_epoch=True)
        self.log('jaccard_value', jaccard_value, on_epoch=True)

    def configure_optimizers(self):
        return self.optim


def do_train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        cfg, 
    ):

    my_model = LitModel(model, loss_fn, optimizer, )
    
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
    ):

    my_model = LitModel(model, loss_fn, optimizer, )
    trainer = pl.Trainer(devices=1, accelerator="gpu")
    # ------------
    # testing
    # ------------
    result = trainer.test(model=my_model, test_dataloaders=val_loader)
    print(result)