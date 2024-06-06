from utils import Constants
from pathlib import Path
from enum import Enum
from utils import Constants
from pathlib import Path
from train.models.train_parameters import TrainParameters
from abc import ABC, abstractmethod


class TrainConstants:
    train_dataset_root = Path("./datasets_for_train")

class TrainModel(ABC):

    @abstractmethod
    def loadModel(self,model_path: Path):
        ...
    
    @abstractmethod
    def trainModel(self,train_parameters: TrainParameters):
        ...
    
    @abstractmethod
    def exportModel(self,dest_path: Path):
        ...
    
    @abstractmethod
    def evalModel(self,):
        ...

class Comet:
    parameters = {
        Constants.GaugeType.digital.value: {
            "api_key": "GrkgXtu7qg8v5K6kgx6Epn8t6", #TODO: the same as all project
            "project_name": "digital",
            "workspace": "abc-gauge-detection",
        },
        Constants.GaugeType.dial.value: {
            
        },
        Constants.GaugeType.number.value: {
            "api_key": "GrkgXtu7qg8v5K6kgx6Epn8t6", #TODO: the same as all project
            "project_name": "number",
            "workspace": "abc-gauge-detection",
        },
        Constants.GaugeType.clock.value: {
            
        },
        Constants.GaugeType.level.value: {
            
        }
    }
