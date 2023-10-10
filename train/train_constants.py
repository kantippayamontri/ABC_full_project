from utils import Constants
from pathlib import Path
from enum import Enum
from utils import Constants
class TrainConstants:
    train_dataset_root = Path("./datasets_for_train")
    
    __train_digital_dataset_path = train_dataset_root / Constants.GaugeType.digital.value
    __train_dial_dataset_path = train_dataset_root / Constants.GaugeType.dial.value
    __train_number_dataset_path = train_dataset_root / Constants.GaugeType.number.value
    __train_level_dataset_path = train_dataset_root / Constants.GaugeType.level.value
    __train_clock_dataset_path = train_dataset_root / Constants.GaugeType.clock.value
    
    train_dataset_path_dict = {
        Constants.GaugeType.digital: __train_digital_dataset_path,
        Constants.GaugeType.dial: __train_dial_dataset_path,
        Constants.GaugeType.number: __train_number_dataset_path,
        Constants.GaugeType.clock: __train_clock_dataset_path,
        Constants.GaugeType.level: __train_level_dataset_path
    }
    

class Comet:
    parameters = {
        Constants.DatasetType.TYPE_3 : {
            Constants.DatasetUse.TYPE_3_GAUGE_DISPLAY_FRAME : {
                "api_key": "GrkgXtu7qg8v5K6kgx6Epn8t6",
                "project_name":"abc-digital-gauge",
                "workspace":"undefined",
            },
        },
    }