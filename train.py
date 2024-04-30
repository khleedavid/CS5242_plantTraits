from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from simple_cnn import SimpleCNN
from minimal_lightning import MultipleInputRegressions
import lightning as L
from lightning.pytorch import Trainer
from image_dataset import ImageDataset
from torch.utils.data import DataLoader, random_split

data_path = "./data/planttraits2024_1"


def setup():
    image_data = ImageDataset(csv_file_path=f"{data_path}/train.csv", image_dir=f"{data_path}/train_images")
    train_size = int(0.80 * len(image_data))
    val_size = int((len(image_data) - train_size) / 2)
    test_size = int((len(image_data) - train_size) / 2)

    train_set, val_set, test_set = random_split(image_data, (train_size, val_size, test_size))
    return train_set, val_set, test_set

if __name__ == "__main__":
    train_set, val_set, test_set = setup()
    model = MultipleInputRegressions()
    train_loader = DataLoader(train_set, 32)
    val_loader = DataLoader(val_set, 32)
    test_loader = DataLoader(test_set, 32)
    trainer = L.Trainer(max_epochs=5)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)

