from typing import Any
from torchvision import transforms
from torch import nn
import torch
import timm
import lightning as L
from torchmetrics.regression import R2Score
from torch.utils.data import DataLoader
from config import Config

CONFIG = Config()

class Model(L.LightningModule):
    def __init__(self, lr: float = 1e-3, num_workers: int = 4, batch_size: int =32):
        super().__init__()
        self.lr = lr
        # ImageNet Normalize Input
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # Backbone
        self.backbone = timm.create_model(
                'efficientvit_b1.r288_in1k',
                pretrained=True,
                num_classes=0,
            )
        
        # Features
        self.features = nn.Sequential(
            nn.Linear(CONFIG.N_FEATURES,256),
            nn.GELU(),
            nn.Linear(256,256),
        )
        
        # Label
        self.label = nn.Sequential(
            nn.Linear(256,256),
            nn.GELU(),
            nn.Linear(256,CONFIG.N_TARGETS, bias=False),
        )
        
        # Initialize Weights
        self.initialize_weights()
        
    def initialize_weights(self):
        # Features
        nn.init.kaiming_uniform_(self.features[2].weight)
        # Label
        nn.init.zeros_(self.label[2].weight)
        
    def forward(self, inputs, debug=False):
        y_pred = self.backbone(self.normalize(inputs['image'].float() / 255)) + self.features(inputs['feature'])
        return y_pred
    
    def training_step(self, batch):
        X_sample, y_sample = batch
        criterion = R2Score()
        y_pred = self(X_sample)
        loss = criterion(y_pred, y_sample)
        print(loss)
        return loss
    
    def on_train_epoch_end(self):
        all_preds = torch.stack(self.train_step_outputs)

    def validation_step(self, batch, batch_idx):
        X_sample, y_sample = batch
        criterion = R2Score()
        y_pred = self(X_sample)
        loss = criterion(y_pred, y_sample)
        print(loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr))
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size = self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size = self.batch_size)