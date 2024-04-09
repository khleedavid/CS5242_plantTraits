import lightning as L
import torch.nn as nn
import torch
import warnings
import pandas as pd

warnings.simplefilter(action="ignore", category=UserWarning)

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)),
        nn.ReLU(),
        nn.BatchNorm2d(output_size),
        nn.MaxPool2d((2, 2))
    )
    return block

class MultipleInputRegressions(L.LightningModule):
    def __init__(
            self, lr: float = 1e-3, num_workers: int = 4, batch_size: int = 32
    ):
        super().__init__()
        self.lr = lr
        self.val_step_outputs = []
        self.train_step_outputs = []
        self.train_step_losses = []
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

        self.ln4 = nn.Linear(163, 100)
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
        
        y_pred = y_pred.float()

        loss = criterion(y_pred, y)
        self.train_step_outputs.append(y_pred)
        self.train_step_losses.append(loss)
        self.log("loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        image, tabular, y = batch
        criterion = torch.nn.L1Loss()
        y_pred = self(image, tabular)
        # y_pred = torch.flatten(self(image, tabular))
        y_pred = y_pred.float()

        test_loss = criterion(y_pred, y)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        image, tabular, y = batch
        
        criterion = torch.nn.L1Loss()
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
    