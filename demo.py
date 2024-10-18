import os
import shutil
import kagglehub
import yaml
from load_agro_dataset import load_agro_dataset
import pandas as pd
from src.models.crop_tarnsformer import CropTransformer
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import random

def load_model(filename):
    """
        Загружает модель из kaggle.com по пути filename, если она уже не загружена.
    """
    if not os.path.exists(filename):
        folder_path = os.path.dirname(filename)
        os.makedirs(folder_path, exist_ok=True)

        path = kagglehub.model_download("danildolgov/crop-transformer/pyTorch/default")
        cache_file = os.listdir(path)[0]
        
        shutil.move(os.path.join(path, cache_file), filename)
    
    return filename


if __name__ == "__main__":
    # Конфиг, который содержит путь до датасета
    with open('configs/train.yaml') as file:
        train_config = yaml.safe_load(file)

    # Если датасет не иницилизирован, он загружается с сервера
    if not os.path.exists(train_config["crop_dataset_path"]):
        load_agro_dataset(train_config["crop_dataset_path"])

    # Загрузка датасета
    df = pd.read_csv(train_config["crop_dataset_path"])

    date = pd.to_datetime(df['date'])
    df["hour"] = date.dt.hour
    df["day"] = date.dt.day
    df["month"] = date.dt.month
    df.drop(columns=['date'], inplace=True)
    
    # После преобразования датасет имеет следующие колонки:
    # colomns = [
    #   SOLAR_RADIATION, 
    #   PRECIPITATION, 
    #   WIND_SPEED, 
    #   LEAF_WETNESS, 
    #   HC_AIR_TEMPERATURE, 
    #   HC_RELATIVE_HUMIDITY, 
    #   DEW_POINT,
    #   hour,
    #   day,
    #   month
    # ]

    # Каждая строка датасета отражает данные за 1 час. 
    # Будем предсказывать 168 часов на основе 168 строк.
    count_generations = 168 # 168 часов = 7 дней
    context_size = count_generations # 168 часов = 7 дней
    
    # Взятие срезов данных
    random_point = random.randint(0, len(df) - context_size) # Немного рандома
    full_x = df.iloc[random_point:random_point + context_size + count_generations].to_numpy() # Контекст с истинным продолжением
    context_x = full_x[:context_size] # Конеткст, context_x.shape = (context_size, len(df.columns)) = (168, 10)
    
    # Загрузка модели. P.S. Метод load_model() используется для автозагрузки
    # модели с Kaggle. Если в этом нет необходимости, можно просто использовать путь до модели
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model:CropTransformer = CropTransformer.from_checkpoint(load_model("models/crop-transformer/model.pt"), device)[0]
    
    # Генерация предсказания. pred представляет собой обычный NumPy массив, 
    # в нашем случае размером pred.shape = (count_generations, len(df.columns)) = (168, 10)
    pred = model.predict(context_x, count_generations)
    
    # Отображение резульата
    start_hour = (pd.Timestamp(hour=int(full_x[0, -3]), day=int(full_x[0, -2]), month=int(full_x[0, -1]), year=2024) - pd.Timestamp("01-01-2024")).total_seconds()/3600
    h = np.arange(start_hour, start_hour + context_size + count_generations)

    n_cols = 2
    n_rows = math.ceil(len(df.columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    axes = axes.flatten()
    for ax in axes:
        ax.axis('off')
    
    for i, (ax, parameter_name) in enumerate(zip(axes, df.columns)):
        ax.plot(h, full_x[:, i], label="y")
        ax.plot(h[-count_generations:], pred[:, i], marker='o', markevery=[0], linestyle='--', label="predicted y")
        ax.legend()
        ax.axis('on')
        ax.set_xlabel("hour")
        ax.set_ylabel(parameter_name)
    
    plt.tight_layout()
    plt.show()
    