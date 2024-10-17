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

# weather_parameters = ["SOLAR_RADIATION",
#                         "PRECIPITATION",
#                         "WIND_SPEED",
#                         "LEAF_WETNESS",
#                         "HC_AIR_TEMPERATURE",
#                         "HC_RELATIVE_HUMIDITY",
#                         "DEW_POINT"]

class TrainParameters():
    def __init__(self, context_size, batch_size, epochs, learning_rate, 
                 saving_freq, save_path, validation_freq, count_validation_steps, generation_freq):
        self.context_size = context_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.saving_freq = saving_freq
        self.save_path = save_path
        self.validation_freq = validation_freq
        self.count_validation_steps = count_validation_steps
        self.generation_freq = generation_freq


def get_validation_loss(model: CropTransformer, val_dataset: DataLoader, device_manager: DeviceManager, count_validation_steps:int=-1):
    """
    Оцениват значение функции потерь на валидационной выборке.
    
    **Если count_validation_steps < 0, тогда валидация будет проходить 1 эпоху**
    """
    with torch.no_grad():
        model.eval()
        count_validation_steps = len(val_dataset) if count_validation_steps < 0 else count_validation_steps
        validation_step = 0
        val_loss = torch.tensor(0.0, dtype=torch.float32, device=device_manager.device)
        for x, y in val_dataset:
            x, y = x.to(device_manager.device), y.to(device_manager.device)
            x, y = model.input_scaler(x), model.input_scaler(y)

            y_pred = model(x)
            loss = nn.MSELoss()(y_pred, y)
            val_loss += loss / count_validation_steps

            if validation_step >= count_validation_steps:
                break
        model.train()

    return val_loss.cpu()


def generate_predictions(model: CropTransformer, val_loader: DataLoader, parameter_names: List[str], save_dir: str, comet_manager: CometManager, device_manager: DeviceManager):
    with torch.no_grad():
        x, _ = next(iter(val_loader))
        x = x[0][None].to(device_manager.device) # (1, T, C)
        
        context_part_size = x.size(1) // 2
        context_x = x[:, :context_part_size]

        pred_right_x = model.generate(context_x, x.size(1) - context_part_size)
        cpred_right_x = torch.cat((context_x[:, -1:], pred_right_x), dim=1)
        x, cpred_right_x = x.cpu(), cpred_right_x.cpu()
        

        count_date_intervals = model.count_date_intervals.item()
        size_interval = 366 * 24 / count_date_intervals
        h = torch.sum(x[..., -count_date_intervals:], dim=-1)[0] * size_interval

        predict_parameter_names = parameter_names[:-count_date_intervals]

        n_cols = 2
        n_rows = math.ceil(len(predict_parameter_names) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
        axes = axes.flatten()
        for ax in axes:
            ax.axis('off')
        
        for i, (ax, parameter_name) in enumerate(zip(axes, predict_parameter_names)):
            ax.plot(h.numpy(), x[0, :, i].numpy(), label="y")
            ax.plot(h[-cpred_right_x.size(1):].numpy(), cpred_right_x[0, :, i].numpy(), marker='o', markevery=[0], linestyle='--', label="predicted y")
            ax.legend()
            ax.axis('on')
            ax.set_xlabel("hour")
            ax.set_ylabel(parameter_name)
        
        plt.tight_layout()

        image_path = os.path.join(save_dir, "predictions.png")
        plt.savefig(image_path, dpi=300)
        comet_manager.log_image(image_path, "predictions")


def load_comet_data(use_comet: bool):
    if use_comet:
        if not os.path.exists(".env"):
            print("У вас не инициализированны параметры окружения comet_ml. Хотите использовать comet_ml? (y/n)")
            if input() == "y":
                print("Введите параметры окружения comet_ml:")
                comet_api_key = input("COMET_API_KEY: ")
                comet_project_name = input("COMET_PROJECT_NAME: ")
                comet_workspace = input("COMET_WORKSPACE: ")
                
                with open(".env", "w") as f:
                    f.write(f"COMET_API_KEY={comet_api_key}\n")
                    f.write(f"COMET_PROJECT_NAME={comet_project_name}\n")
                    f.write(f"COMET_WORKSPACE={comet_workspace}\n")

                return comet_api_key, comet_project_name, comet_workspace, True
        else:
            load_dotenv()
            comet_api_key = os.getenv("COMET_API_KEY")
            comet_project_name = os.getenv("COMET_PROJECT_NAME")
            comet_workspace = os.getenv("COMET_WORKSPACE")
            return comet_api_key, comet_project_name, comet_workspace, True
    return "comet_not_use", "comet_not_use", "comet_not_use", False


def run(device_manager: DeviceManager, train_parameters: TrainParameters, comet_manager: CometManager):
    train_dataset, val_dataset = CropDataset.get_train_and_valid("data/argo_dataset/argo_dataset.csv", 
                                                                 context_size=train_parameters.context_size, 
                                                                 num_aug_copies=5,
                                                                 count_date_intervals=5)
    train_loader = DataLoader(train_dataset, train_parameters.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, train_parameters.batch_size, shuffle=True)

    config = CropTransformerConfig()
    model = CropTransformer(config).init_scaler(train_dataset.mean, train_dataset.std)
    model = model.to(device_manager.device)

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=train_parameters.learning_rate)

    saver = Saver(train_parameters.save_path, device_manager)

    total_step = 0
    validation_step = 0
    for epoch in range(train_parameters.epochs):
        t0 = time.time()
        step = 0
        for x, y in train_loader:
            x, y = x.to(device_manager.device), y.to(device_manager.device)
            x, y = model.input_scaler(x), model.input_scaler(y)
            
            model.train()
            optimizer.zero_grad()
            preds = model(x)
            loss = nn.MSELoss()(preds, y)
            loss.backward()
            optimizer.step()
            
            is_save_step = total_step > 0 and total_step % train_parameters.saving_freq == 0
            is_validation_step = is_save_step or (total_step > 0 and total_step % train_parameters.validation_freq == 0)
            is_generation_step = total_step % train_parameters.generation_freq == 0

            if is_validation_step:
                val_loss = get_validation_loss(model, val_loader, device_manager, train_parameters.count_validation_steps).item()

            if is_save_step:
                print("Saving model...")
                saver.save(model, optimizer, val_loss)
            
            if is_generation_step:
                print("Generating predictions...")
                generate_predictions(model, val_loader, train_dataset.parameter_names, train_parameters.save_path, comet_manager, device_manager)

            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            val_loss_str = str(val_loss) if is_validation_step else "-"
            log_line = f"total_step {total_step}, epoch {epoch}, step {step}  | loss: {loss.item()} | dt: {dt:.3f} | val_loss: {val_loss_str}"
            print(log_line)
            saver.save_log(log_line)

            if comet_manager.use_comet:
                comet_manager.experiment.log_metric("loss", loss.item(), step=total_step)
                comet_manager.experiment.log_metric("dt", dt, step=total_step)
                if is_validation_step:
                    comet_manager.experiment.log_metric("val_loss", val_loss, step=total_step, epoch=validation_step)

            total_step += 1
            step += 1
            validation_step += int(is_validation_step)
            t0 = time.time()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random_seed = 1337 # Для получения детерменированных результатов
    context_size = 128
    batch_size = 16
    epochs = 100000
    learning_rate = 1e-6
    saving_freq = 1000
    validation_freq = 100
    generation_freq = 100
    count_validation_steps = 5
    save_path = "models\checkpoint_model"

    use_comet = True
    comet_api_key, comet_project_name, comet_workspace, use_comet = load_comet_data(use_comet)

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    device_manager = DeviceManager(device)
    train_parameters = TrainParameters(context_size, batch_size, epochs, learning_rate, saving_freq, 
                                       save_path, validation_freq, count_validation_steps, generation_freq)
    comet_manager = CometManager(comet_api_key, comet_project_name, comet_workspace, device_manager, use_comet=use_comet)

    run(device_manager, train_parameters, comet_manager)