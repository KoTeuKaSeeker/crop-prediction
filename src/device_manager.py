import torch

class DeviceManager():
    """
    Пока что этот класс выглядит избыточным, но в последующем
    его модификация очень сильно пригодится, если появится необходимость
    использовать распределённые вычисления на нескольких GPU или TPU.
    """
    def __init__(self, device):
        self.device = device
    

    def mark_step(self):
        """
        Синхронизирует ускоритель с cpu.
        """

        if torch.cuda.is_available():
            torch.cuda.synchronize()