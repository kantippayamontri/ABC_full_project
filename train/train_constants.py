from utils import Constants
from pathlib import Path
from enum import Enum
from utils import Constants


class TrainConstants:
    train_dataset_root = Path("./datasets_for_train")

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
