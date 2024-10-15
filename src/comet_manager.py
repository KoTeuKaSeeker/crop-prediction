from comet_ml import Experiment
from src.device_manager import DeviceManager
import requests
import json

class CometManager():
    def __init__(self, api_key: str, project_name: str, workspace: str, device_manager: DeviceManager, use_comet=True):
        self.use_comet = use_comet
        self.api_key = api_key
        self.project_name = project_name
        self.workspace = workspace
        self.device_manager = device_manager

        if use_comet:
            self.experiment = Experiment(api_key, project_name, workspace)
        else:
            self.experiment = None
    
    def log_image(self, image_path: str, name: str):
        if self.use_comet:
            self.experiment.log_image(image_path, name=name)