import pandas as pd
import numpy as np
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

warnings.simplefilter(action="ignore", category=FutureWarning)
data_path = "./data/planttraits2024_1/"
target_columns = ["X4_mean", "X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean", "X4_sd", "X11_sd", "X18_sd", "X26_sd", "X50_sd", "X3112_sd"]

class ImageDataset(Dataset):
    """Combining tabular data and image data into a single dateset."""

    def __init__(self, csv_file_path, image_dir):
        self.image_dir = image_dir
        plant_df = pd.read_csv(csv_file_path)
        ## should remove the na data first
        self.plant_df = clean_up_na_data(plant_df)
        print(self.plant_df.shape)

    def __len__(self):
        return len(self.plant_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        plant_traits = self.plant_df.iloc[idx]
        y = torch.FloatTensor(plant_traits[target_columns])
        image_id = int(plant_traits['id'])
        image = Image.open(f"{self.image_dir}/{image_id}.jpeg")
        image = np.array(image)
        
        image = transforms.functional.to_tensor(image)

        plant_traits = plant_traits.drop(target_columns)
        plant_traits = plant_traits.tolist()[1:]
        plant_traits = torch.FloatTensor(plant_traits)
        return image, plant_traits, y

def clean_up_na_data(df):
    return df.dropna()

class TestImageDataset(Dataset):
    def __init__(self, csv_file_path, image_dir):
        self.image_dir = image_dir
        plant_df = pd.read_csv(csv_file_path)
        ## should remove the na data first
        self.plant_df = clean_up_na_data(plant_df)

    def __len__(self):
        return len(self.plant_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        plant_traits = self.plant_df.iloc[idx]
        image_id = int(plant_traits['id'])
        image = Image.open(f"{self.image_dir}/{image_id}.jpeg")
        image = np.array(image)
        
        image = transforms.functional.to_tensor(image)

        plant_traits = plant_traits.tolist()
        plant_traits = torch.FloatTensor(plant_traits[1:])
        return image_id, image, plant_traits