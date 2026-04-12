import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from src.models.vgg19 import VGGModel
from src.models.lit_vgg import LitVGG
torch.set_float32_matmul_precision('high')

def simple_train(model, train_dataloader, val_dataloader, epochs=10):
    """
    Train Model using Pytorch Lightning Framework to simplify training code
    """

    tb_logger = TensorBoardLogger(save_dir="logs/", name="my_model")
    lr_callback = LearningRateMonitor(logging_interval="step")
    loggers = [tb_logger]
    callbacks = [lr_callback]
    try:
        wandb_logger = WandbLogger(project = "deep_learning_project", log_model="all")
        loggers.append(wandb_logger)
    except Exception:
        print("Wandb not available, logging to Tensorboard only.")

    trainer = Trainer(
        max_epochs=epochs, 
        accelerator="auto", 
        logger=loggers,
        callbacks=callbacks
    )
    trainer.fit(model, train_dataloader, val_dataloader, epochs=epochs)

if __name__=="__main__":

    model = VGGModel()
    model = LitVGG(model)

    data_transforms = torchvision.models.VGG19_BN_Weights.DEFAULT.transforms()
    
    train_dir = "data/train"
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    val_dir = "data/val"
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )

    simple_train(model, train_dataloader, val_dataloader)