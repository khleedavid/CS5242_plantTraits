import torch
import torch.nn as nn
import pytorch_lightning as pl
import lightning as L
from torch.utils.data import DataLoader, random_split

from image_dataset import ImageDataset
data_path = "./data/planttraits2024"


def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)),
        nn.ReLU(),
        nn.BatchNorm2d(output_size),
        nn.MaxPool2d((2, 2))
    )
    return block

class SimpleCNN(L.LightningModule):
    def __init__(
            self, lr: float = 1e-3, num_workers: int = 4, batch_size: int = 32
    ):
        super().__init__()
        self.lr = lr
        self.val_step_outputs = []
        self.train_step_outputs = []
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.conv1 = conv_block(3, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)

        self.ln1 = nn.Linear(64 * 62 * 62, 16)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout2d(0.5)
        self.ln2 = nn.Linear(16, 12)

        self.ln4 = nn.Linear(164, 100)
        self.ln5 = nn.Linear(100, 50)
        self.ln6 = nn.Linear(50, 12)

        self.ln7 = nn.Linear(24, 12)

    def forward(self, img, tab):
        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        img = img.reshape(img.shape[0], -1)
        img = self.ln1(img)
        
        img = self.relu(img)
        img = self.batchnorm(img)
        img = self.dropout(img)
       
        img = self.ln2(img)
        img = self.relu(img)
        

        tab = self.ln4(tab)
        tab = self.relu(tab)
        tab = self.ln5(tab)
        tab = self.relu(tab)
        tab = self.ln6(tab)
        tab = self.relu(tab)
        

        x = torch.cat((img, tab), dim=1)
        x = self.relu(x)
        y_pred = self.ln7(x)
        return y_pred
    
    def training_step(self, batch, batch_idx):
        image, tabular, y = batch
        criterion = torch.nn.L1Loss()
        y_pred = self(image, tabular)
        
        # y_pred = torch.flatten(y_pred)
        y_pred = y_pred.float()

        loss = criterion(y_pred, y)

        tensorboard_logs = {"train_loss": loss}
        self.train_step_outputs.append({"loss": loss, "log": tensorboard_logs})
        # return {"loss": loss, "log": tensorboard_logs}
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, tabular, y = batch
        
        criterion = torch.nn.L1Loss()
        y_pred = self(image, tabular)
        
        y_pred = y_pred.float()
        

        val_loss = criterion(y_pred, y)
        self.val_step_outputs.append({"val_loss": val_loss})
        # return {"val_loss": val_loss}
        return val_loss
    
    # def on_validation_epoch_end(self):
    #     avg_loss = torch.stack([x["val_loss"] for x in self.val_step_outputs]).mean()
    #     tensorboard_logs = {"val_loss": avg_loss}
    #     return {"val_loss": avg_loss, "log": tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        image, tabular, y = batch
        criterion = torch.nn.L1Loss()
        y_pred = self(image, tabular)
        # y_pred = torch.flatten(self(image, tabular))
        y_pred = y_pred.float()

        test_loss = criterion(y_pred, y)

        # return {"test_loss": test_loss}
        return test_loss
    
    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    #     logs = {"test_loss": avg_loss}
    #     return {"test_loss": avg_loss, "log": logs, "progress_bar": logs}

    def setup(self, stage):
        image_data = ImageDataset(csv_file_path=f"{data_path}/train.csv", image_dir=f"{data_path}/train_images")
        train_size = int(0.80 * len(image_data))
        val_size = int((len(image_data) - train_size) / 2)
        test_size = int((len(image_data) - train_size) / 2)

        self.train_set, self.val_set, self.test_set = random_split(image_data, (train_size, val_size, test_size))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr))
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size = self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size = self.batch_size)
    