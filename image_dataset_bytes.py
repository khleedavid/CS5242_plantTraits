import pandas as pd
import numpy as np
import warnings
import imageio.v3 as imageio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


from config import Config

warnings.simplefilter(action="ignore", category=FutureWarning)
data_path = "./data/planttraits2024_1/"
target_columns = ["X4_mean", "X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean", "X4_sd", "X11_sd", "X18_sd", "X26_sd", "X50_sd", "X3112_sd"]
LOG_FEATURES = ['X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
CONFIG = Config()
FEATURE_SCALER = StandardScaler()

# Mask to exclude values outside of 0.1% - 99.9% range
def get_mask(df):
    lower = []
    higher = []
    mask = np.empty(shape=df[CONFIG.TARGET_COLUMNS].shape, dtype=bool)
    # Fill mask based on minimum/maximum values of sample submission
    for idx, (t, v_min, v_max) in enumerate(zip(CONFIG.TARGET_COLUMNS, CONFIG.V_MIN, CONFIG.V_MAX)):
        labels = df[t].values
        mask[:,idx] = ((labels > v_min) & (labels < v_max))
    return mask.min(axis=1)

# Fill labels using normalization tool
def fill_y(y, df, Y_SHIFT, Y_STD, normalize=False):
    
    for target_idx, target in enumerate(CONFIG.TARGET_COLUMNS):
        v = df[target]
        if normalize:
            # Log10 Transform
            if target in LOG_FEATURES:
                v = np.log10(v)
            # Shift To Have Zero Median
            Y_SHIFT[target_idx] = np.mean(v)
            v = v - np.median(v)
            # Uniform Variance
            Y_STD[target_idx] = np.std(v)
            v = v / np.std(v)
        # Assign to y_train
        y[:,target_idx] = v

transformation = A.Compose([
        A.Resize(CONFIG.IMAGE_SIZE,CONFIG.IMAGE_SIZE),
        ToTensorV2(),
    ])

class ImageDataset(Dataset):
    """Combining tabular data and image data into a single dateset."""

    def __init__(self, csv_file_path, image_dir):
        self.image_dir = image_dir
        plant_df = pd.read_csv(csv_file_path)
        # add image path
        plant_df['file_path'] = plant_df['id'].apply(lambda s: f'{image_dir}/{s}.jpeg')
        plant_df['jpeg_bytes'] = plant_df['file_path'].apply(lambda fp: open(fp, 'rb').read())
        # assign median
        CONFIG.TARGET_MEDIANS = plant_df[CONFIG.TARGET_COLUMNS].median(axis=0).values
        # filter out the outliers
        CONFIG.V_MIN = plant_df[CONFIG.TARGET_COLUMNS].quantile(0.001)
        CONFIG.V_MAX = plant_df[CONFIG.TARGET_COLUMNS].quantile(0.999)
        plant_df[CONFIG.TARGET_COLUMNS].quantile(0.001)
        mask = get_mask(plant_df)
        masked_train = plant_df[mask].reset_index(drop=True)
        # Fill labels using normalization tool
        Y_SHIFT = np.zeros(CONFIG.N_TARGETS)
        Y_STD = np.zeros(CONFIG.N_TARGETS)
        y_train_mask_raw = np.zeros_like(masked_train[CONFIG.TARGET_COLUMNS], dtype=np.float32)
        y_train_mask = np.zeros_like(masked_train[CONFIG.TARGET_COLUMNS], dtype=np.float32)
        fill_y(y_train_mask_raw, masked_train, Y_SHIFT, Y_STD, normalize=False)
        fill_y(y_train_mask, masked_train,Y_SHIFT, Y_STD, normalize=True)
        # features scaling
        train_features_mask = FEATURE_SCALER.fit_transform(masked_train[CONFIG.FEATURE_COLUMNS].values.astype(np.float32))
        self.plant_df = train_features_mask
        self.images = masked_train["jpeg_bytes"].values
        self.y = y_train_mask


    def __len__(self):
        return len(self.plant_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X_sample = {
            'image': transformation(image=imageio.imread(self.images[idx]))['image'],
            'feature': self.plant_df[idx]
        }
        y_sample = self.y[idx]
        return X_sample, y_sample





