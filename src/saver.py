from src.device_manager import DeviceManager
import torch
import torch.nn as nn
import os

class Saver():
    def __init__(self, checkpoint_dir: str, device_manager: DeviceManager):
        self.device_manager = device_manager
        self.checkpoint_dir = checkpoint_dir
        self._best_val_loss = None

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.reset_log()
    
    def save_checkpoint(self, filename: str, model: nn.Module, optimizer: torch.optim.Optimizer, val_loss):
        """
        Сохраняет только один чекпоинт
        """
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, filename)


    def reset_log(self):
        """
        Очищает содержание файла логов
        """

        log_path = os.path.join(self.checkpoint_dir, "log.txt")
        with open(log_path, "w") as f:
            pass


    def save_log(self, log_line: str):
        """
        Добавляет строчку log_line в файл логов обучения модели. Файл логов может
        использовать для отображения процесса обучения модели в виде графика после
        завершения обучения.
        """

        log_path = os.path.join(self.checkpoint_dir, "log.txt")
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

    
    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer, val_loss):
        """
        Сохраняет последную версию модели (last) и лучшую версию модели (best)
        """
        
        last_model_path = os.path.join(self.checkpoint_dir, "last.pt")
        self.save_checkpoint(last_model_path, model, optimizer, val_loss)

        if self._best_val_loss is None or val_loss <= self._best_val_loss:
            self._best_val_loss = val_loss
            best_model_path = os.path.join(self.checkpoint_dir, "best.pt")
            self.save_checkpoint(best_model_path, model, optimizer, val_loss)