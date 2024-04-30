from image_dataset import ImageDataset
from minimal_lightning import MultipleInputRegressions
import lightning as L
import pandas as pd
import torch
from torchmetrics.regression import R2Score

data_path = "./data/planttraits2024_1"
from torch.utils.data import DataLoader, random_split

target_columns = ["X4_mean", "X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean", "X4_sd", "X11_sd", "X18_sd", "X26_sd", "X50_sd", "X3112_sd"]

torch

def setup():
    image_data = ImageDataset(csv_file_path=f"{data_path}/train.csv", image_dir=f"{data_path}/train_images")
    train_size = int(0.80 * len(image_data))
    val_size = int((len(image_data) - train_size) / 2)
    test_size = int((len(image_data) - train_size) / 2)

    train_set, val_set, test_set = random_split(image_data, (train_size, val_size, test_size))
    return train_set, val_set, test_set

if __name__ == "__main__":
    train_set, val_set, test_set = setup()
    model = MultipleInputRegressions.load_from_checkpoint("lightning_logs/version_25/checkpoints/epoch=2-step=2937.ckpt")
    trainer = L.Trainer()
    tabular = []
    ys = []
    for idx in range(len(val_set)):
        image, plant_traits, y = val_set[idx]
        tabular.append(plant_traits)
        ys.append(y)
    tabular_whole = torch.stack(tabular)
    ys_whole = torch.stack(ys)

    model.eval()

    # predictions = trainer.predict(model, val_loader)
    # final_outputs = pd.concat(predictions, axis=0)
    # y_pred = torch.tensor(final_outputs.values)
    # y = torch.tensor(val_set[target_columns].values)

    # r2 = R2Score()
    # print(r2(y_pred, y))
    

    print(tabular_whole.size())
    print(ys_whole.size())

