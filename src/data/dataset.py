import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import wandb
from loguru import logger

from peft import LoraConfig # type: ignore
from PIL import Image
from torch.utils.data import Dataset
# from torchvision import transforms

class PlaneTypeDataset(Dataset):
    def __init__(self, 
                 csv_file: str = os.path.join(os.getcwd(), 'data', 'data.csv'), 
                 root_dir: str = os.path.join(os.getcwd(), 'data', 'resized_images'), 
                 processor: Optional[nn.Module] = None, 
                 transform: None = None) -> None:
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(), label