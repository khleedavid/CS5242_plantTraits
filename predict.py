from minimal_lightning import MultipleInputRegressions
from image_dataset import TestImageDataset
from torch.utils.data import DataLoader
import pandas as pd
import lightning as L

data_path = "./data/planttraits2024_1"


if __name__ == "__main__":
    data = TestImageDataset(csv_file_path=f"{data_path}/test.csv", image_dir=f"{data_path}/test_images")
    data_loader = DataLoader(data, 32)
    model = MultipleInputRegressions.load_from_checkpoint("lightning_logs/version_73/checkpoints/epoch=4-step=4895.ckpt")
    trainer = L.Trainer()

    model.eval()
    
    predictions = trainer.predict(model, data_loader)
    
    final_outputs = pd.concat(predictions, axis=0)
    print(final_outputs.shape)
    final_outputs.to_csv("predictions.csv")