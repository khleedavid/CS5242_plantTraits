import lightning as L
import torch.nn as nn
import torch
import pandas as pd

class MultipleInputVGGRegression(L.LightningModule):
    def __init__(
            self, lr: float=1e-3, num_workers: int = 4, batch_size: int = 32
    ):
        super().__init__()
        self.lr = lr
        self.val_step_outputs = []
        self.train_step_outputs = []
        self.train_step_losses = []
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 1 maxpool layer
            nn.MaxPool2d(kernel_size=2, stride=2),


            # 2nd block
            # 2 convolution layers
            nn.Conv2d(64, 128, kernel_size=3, padding =1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 1 maxpool layer
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 3rd block
            # 3 convolution layers
            nn.Conv2d(128, 256, kernel_size=3, padding =1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            # 1 maxpool layer
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 4th block
            # 3 convolution layers
            nn.Conv2d(256, 512, kernel_size=3, padding =1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            # 1 maxpool layer
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 5th block
            # 3 convolution layers
            nn.Conv2d(512, 512, kernel_size=3, padding =1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            # 1 maxpool layer
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(131072, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 12),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(163, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 12),
            nn.ReLU(inplace=True)
        )

        self.final_linear = nn.Linear(24, 12)

    def forward(self, img, tab):
        img = self.conv_layers(img)
        # unroll the image
        img = img.reshape(img.shape[0], -1)
        img = self.fc_layers(img)

        tab = self.linear_layers(tab)

        # concat both data
        x = torch.cat((img, tab), dim=1)
        y_pred = self.final_linear(x)
        return y_pred
    
    def training_step(self, batch, batch_idx):
        image, tabular, y = batch
        criterion = torch.nn.MSELoss()
        y_pred = self(image, tabular)
        
        y_pred = y_pred.float()

        loss = criterion(y_pred, y)
        self.train_step_outputs.append(y_pred)
        self.train_step_losses.append(loss)
        self.log("training_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        image, tabular, y = batch
        criterion = torch.nn.MSELoss()
        y_pred = self(image, tabular)
        # y_pred = torch.flatten(self(image, tabular))
        y_pred = y_pred.float()

        test_loss = criterion(y_pred, y)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        image, tabular, y = batch
        
        criterion = torch.nn.MSELoss()
        
        y_pred = self(image, tabular)
        
        y_pred = y_pred.float()
        val_loss = criterion(y_pred, y)
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx):
        image_id, image, tabular = batch
        y_pred = self(image, tabular)
        x = ['X4', 'X11', 'X18', 'X50', 'X26', 'X3112']
        outputs = []
        ids = []
        for pred in y_pred:
            samples = []
            for i in range(6):
                sample = torch.normal(pred[i], pred[i+6])
                samples.append(sample)
            outputs.append(samples)
            ids.append(image_id)
        outputs = torch.Tensor(outputs).cpu().numpy()
        ids = pd.Series([id.cpu().numpy() for id in ids][0])
        outputs = pd.DataFrame(outputs, columns=x)
        outputs['id'] = ids
        outputs = outputs.set_index('id')
        return outputs
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr))