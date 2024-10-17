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
import yaml


if __name__ == "__main__":
    model, optimizer, train_config, metrics = CropTransformer.from_checkpoint("test_checkpoint.pt")

    metrics = {"val_loss": 0.01}
    best_metrics = {"val_loss": 0.01}
    model.save("test_run", optimizer, train_config, metrics, best_metrics)
    metrics["val_loss"] = 0.0001
    model.save("test_run", optimizer, train_config, metrics, best_metrics)
    print(best_metrics)


    model, optimizer, train_config, metrics = CropTransformer.from_checkpoint("test_run/best.pt")

    print("Hi hi")
    
    
    # model = CropTransformer.from_config("configs/crop_transformer.yaml")

    # optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=0.1)

    # with open("configs/train.yaml", "r") as file:
    #     train_config = yaml.safe_load(file)

    # model.save_checkpoint("test_checkpoint.pt", optimizer, train_config, 0.001)
    

    print("Hello, world!")

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # device_manager = DeviceManager(device)
    # saver = Saver("models/checkpoint_model", device_manager)

    # model, optimizer, val_loss = saver.load("models/checkpoint_model/best.pt")
