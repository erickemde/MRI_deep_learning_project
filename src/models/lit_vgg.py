import lightning as L
from .vgg19 import VGGModel
from torch import optim
from torch import nn
from torchmetrics import Accuracy

class LitVGG(L.LightningModule):
    def __init__(self, model, optimizer = optim.Adam, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = optimizer
        self.train_acc = Accuracy(task="multiclass", num_classes=4)
        self.val_acc = Accuracy(task="multiclass", num_classes=4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.log("train/train_loss", loss)
        self.log("train/train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.log("valid/valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid/valid_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.lr)