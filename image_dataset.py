import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


data_path = "./data/planttraits2024/"
target_columns = ["X4_mean", "X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean", "X4_sd", "X11_sd", "X18_sd", "X26_sd", "X50_sd", "X3112_sd"]

class ImageDataset(Dataset):
    """Combining tabular data and image data into a single dateset."""

    def __init__(self, csv_file_path, image_dir):
        self.image_dir = image_dir
        plant_df = pd.read_csv(csv_file_path)
        ## should remove the na data first
        self.plant_df = clean_up_na_data(plant_df)[:8000]

    def __len__(self):
        return len(self.plant_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        plant_df = self.plant_df.iloc[idx, 0:]
        y = torch.FloatTensor(plant_df[target_columns])
        image_id = int(plant_df['id'])
        image = Image.open(f"{self.image_dir}/{image_id}.jpeg")
        image = np.array(image)
        
        image = transforms.functional.to_tensor(image)

        plant_df = plant_df.drop(target_columns)
        plant_df = plant_df.tolist()
        plant_df = torch.FloatTensor(plant_df)
        return image, plant_df, y

def clean_up_na_data(df):
    return df.dropna()