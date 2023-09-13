import os
from pathlib import Path
from utils.utils import Utils
from enum import Enum, auto


class Constants:
    root_path = Path("./")
    dataset_folder = root_path / "dataset"
    train_dataset_folder = root_path / "dataset" / "train_dataset" # "train_dataset"
    api_key = "aBYrVyIssugbbtJgvuCl"
    model_format = "yolov8"
    project_name = "gauges-detection"
    train_folder = "train"
    val_folder = "valid"
    test_folder = "test"
    image_folder = "images"
    label_folder = "labels"
    setting_yaml = Path("/Users/kantip/Library/Application Support/Ultralytics/settings.yaml")
    dataset_dir_setting = Path("/Users/kantip/Desktop/work/ABC_training") #/dataset/train_dataset/dataset/datasets
    
    # Define the format of data
    class BoundingBoxFormat(Enum):
        YOLOV8 = auto()
        XYXY = auto()

    # Define dataset that want to use to train
    class DatasetUse(Enum):
        TYPE_3_NUMBER = "number"
        TYPE_3_GAUGE_DISPLAY_FRAME = "gauge_display_frame"

    class DatasetType(Enum):
        TYPE_3 = "type3"
    
    class DatasetTrainValTest(Enum):
        TRAIN = auto()
        VAL = auto()
        TEST = auto()

    # Define an enumeration for colors
    colors = {
        0: (255, 0, 0),  # Red
        1: (0, 255, 0),  # Green
        2: (0, 0, 255),  # Blue
        3: (255, 255, 0),  # Yellow
        4: (255, 165, 0),  # Orange
        5: (128, 0, 128),  # Purple
        6: (255, 192, 203),  # Pink
        7: (0, 255, 255),  # Cyan
        8: (255, 0, 255),  # Magenta
        9: (0, 255, 0),  # Lime
        10: (0, 128, 128),  # Teal
        11: (75, 0, 130),  # Indigo
        12: (238, 130, 238),  # Violet
        13: (165, 42, 42),  # Brown
        14: (0, 0, 0),  # Black
        15: (255, 255, 255),  # White
        16: (128, 128, 128),  # Gray
        17: (192, 192, 192),  # Silver
        18: (255, 215, 0),  # Gold
        19: (0, 0, 128),  # Navy
    }

    map_data_dict = {
        "type3": {
            # "gauge": {
            #     "source": ["gauge", "display", "frame"],
            #     "target": ["gauge", "display"],
            # },
            # "frame": {"source": ["display", "frame"], "target": ["display", "frame"], "final_target": ["frame"]},
            DatasetUse.TYPE_3_NUMBER.value: {
                "source": [
                    "dot",
                    "eight",
                    "five",
                    "four",
                    "frame",
                    "nine",
                    "none",
                    "one",
                    "seven",
                    "six",
                    "three",
                    "two",
                    "undefined",
                    "zero",
                ],
                "target": [
                    "dot",
                    "eight",
                    "five",
                    "four",
                    "nine",
                    "none",
                    "one",
                    "seven",
                    "six",
                    "three",
                    "two",
                    "undefined",
                    "zero",
                ],
            },  # remove frame
            DatasetUse.TYPE_3_GAUGE_DISPLAY_FRAME.value: {
                "source": ["gauge", "display", "frame"],
                "target": ["gauge", "display", "frame"],
            },
        }
    }

    dataset_digital = {
        # "gauge": Utils.make_dataset_dict(version=22,
        #                                   api_key=api_key,
        #                                   model_format=model_format,
        #                                   project_name=project_name,
        #                                   dataset_folder=dataset_folder / "type3" / "gauge"),
        # "frame": Utils.make_dataset_dict(
        #     version=24,
        #     api_key=api_key,
        #     model_format=model_format,
        #     project_name=project_name,
        #     dataset_folder=dataset_folder / "type3" / "frame",
        # ),
        DatasetUse.TYPE_3_NUMBER.value: {
            "dataset dict": Utils.make_dataset_dict(
                version=28,
                api_key=api_key,
                model_format=model_format,
                project_name=project_name,
                dataset_folder=dataset_folder / "type3" / "number",
            ),
            "type": DatasetType.TYPE_3.value,
            "key": DatasetUse.TYPE_3_NUMBER.value,
        },
        DatasetUse.TYPE_3_GAUGE_DISPLAY_FRAME.value: {
            "dataset dict": Utils.make_dataset_dict(
                version=27,
                api_key=api_key,
                model_format=model_format,
                project_name=project_name,
                dataset_folder=dataset_folder / "type3" / "gauge_display_frame",
            ),
            "type": DatasetType.TYPE_3.value,
            "key": DatasetUse.TYPE_3_GAUGE_DISPLAY_FRAME.value,
        },
    }
    
    class ModelType(Enum):
        NANO = auto()
        SMALL = auto()
        MEDIUM = auto()
        LARGE = auto()
        EXTRA_LARGE = auto()

    dataset_dial = {}
