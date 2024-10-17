from src.models.crop_tarnsformer import CropTransformer, CropTransformerConfig
from src.crop_dataset import CropDataset
from src.device_manager import DeviceManager
from src.comet_manager import CometManager
from src.saver import Saver
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
import numpy as np
import time
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import math
from typing import List


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_manager = DeviceManager(device)
    saver = Saver("models/checkpoint_model", device_manager)

    model, optimizer, val_loss = saver.load("models/checkpoint_model/best.pt")
