from pathlib import Path
from utils import Constants
class PreprocessConstants:
    base_dataset_folder = Path("./datasets")
    
    # TODO: ration of training set and validation set
    train_ratio = 0.90
    val_ratio = 0.05
    test_ratio = 0.05

    NUMBER_TARGET_SIZE = 640
    NUMBER_TARGET_CLASS = Constants.map_data_dict[Constants.GaugeType.number.value]["target"] 
    
    base_folder_dict = {
        Constants.GaugeType.digital.value : base_dataset_folder / Constants.GaugeType.digital.value,
        Constants.GaugeType.dial.value: base_dataset_folder / Constants.GaugeType.dial.value,
        Constants.GaugeType.number.value: base_dataset_folder / Constants.GaugeType.number.value,
        Constants.GaugeType.clock.value: base_dataset_folder / Constants.GaugeType.clock.value,
        Constants.GaugeType.level.value: base_dataset_folder / Constants.GaugeType.level.value,
    }
    
    train_dataset_folder = Path("./datasets_for_train")
    train_folder_dict = {
        Constants.GaugeType.digital.value : train_dataset_folder / Constants.GaugeType.digital.value,
        Constants.GaugeType.dial.value: train_dataset_folder / Constants.GaugeType.dial.value,
        Constants.GaugeType.number.value: train_dataset_folder / Constants.GaugeType.number.value,
        Constants.GaugeType.clock.value: train_dataset_folder / Constants.GaugeType.clock.value,
        Constants.GaugeType.level.value: train_dataset_folder / Constants.GaugeType.level.value,
    }
    
    