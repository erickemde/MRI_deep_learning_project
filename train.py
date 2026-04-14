import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from src.models.lit_vgg import LitVGG
torch.set_float32_matmul_precision('high')

def simple_train(model, train_dataloader, val_dataloader, epochs=10, run_name="model"):
    """
    Train Model using Pytorch Lightning Framework to simplify training code
    """
    # create callbacks
    lr_callback = LearningRateMonitor(logging_interval="step")
    ckpt_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{run_name}",
        filename=f"{run_name}-epoch={{epoch}}-valid_loss={{valid/valid_loss:.2f}}",
        monitor="valid/valid_loss",
        save_top_k=3,
        mode="min",
        auto_insert_metric_name=False
    )
    callbacks = [lr_callback, ckpt_callback]

    # create loggers
    tb_logger = TensorBoardLogger(save_dir="logs/", name="my_model")
    loggers = [tb_logger]
    try:
        wandb_logger = WandbLogger(project = "deep_learning_project", log_model="all")
        loggers.append(wandb_logger)
    except Exception:
        print("Wandb not available, logging to Tensorboard only.")

    # create training object and fit
    trainer = Trainer(
        max_epochs=epochs, 
        accelerator="auto", 
        logger=loggers,
        callbacks=callbacks
    )
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__=="__main__":

    model = LitVGG()

    data_transforms = model.transforms
    
    # create train dataloader
    train_dir = "data/train"
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )
    # create val dataloader
    val_dir = "data/val"
    val_dataset = datasets.ImageFolder(root=val_dir, transform=data_transforms)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )
    # simple training loop
    simple_train(model, train_dataloader, val_dataloader, run_name="vgg19_bn")