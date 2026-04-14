import lightning.pytorch as pl
import torch
import torch.nn.functional as F

# Conditional import for standalone execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.models.vgg19_attention import VGG19Model
else:
    from .vgg19_attention import VGG19Model


class LitVGG(pl.LightningModule):
    """
    PyTorch Lightning wrapper for VGG19 with Custom Channel Attention
    
    Args:
        use_attention: Enable custom channel attention mechanism
        num_classes: Number of output classes
        learning_rate: Initial learning rate for optimizer
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze VGG19 feature extractor
    """
    def __init__(
        self,
        use_attention=True,
        num_classes=4,
        learning_rate=1e-3,
        pretrained=True,
        freeze_backbone=True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model
        self.model = VGG19Model(
            pretrained=pretrained,
            freeze=freeze_backbone,
            use_attention=use_attention,
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        # Calculate accuracy
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        # Calculate accuracy
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        # Logging
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def get_attention_weights(self):
        """Get attention weights for visualization"""
        return self.model.get_attention_weights()


if __name__ == "__main__":
    # Test Lightning module
    model = LitVGG(use_attention=True, num_classes=4)
    
    # Create dummy batch
    batch = (torch.randn(8, 3, 224, 224), torch.randint(0, 4, (8,)))
    
    # Forward pass
    loss = model.training_step(batch, 0)
    print(f"Training loss: {loss.item():.4f}")
    
    # Get attention weights
    weights = model.get_attention_weights()
    if weights is not None:
        print(f"Attention weights shape: {weights.shape}")