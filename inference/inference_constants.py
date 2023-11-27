from pathlib import Path
from utils import Constants


class InferenceConstants:
    model_root_path = Path("models/")

    model_folder_path_dict = {
        Constants.GaugeType.digital: model_root_path / Path("digital"),
        Constants.GaugeType.number: model_root_path / Path("number"),
        Constants.GaugeType.dial: model_root_path / Path("dial"),
        Constants.GaugeType.clock: model_root_path / Path("clock"),
        Constants.GaugeType.level: model_root_path / Path("level"),
    }

    model_file_path_dict = {
        Constants.GaugeType.digital: model_folder_path_dict[Constants.GaugeType.digital]
        / "digital_model.pt",
        Constants.GaugeType.number: model_folder_path_dict[Constants.GaugeType.number]
        / "number_model.pt",
        Constants.GaugeType.dial: model_folder_path_dict[Constants.GaugeType.dial]
        / "dial_model.pt",
        Constants.GaugeType.clock: model_folder_path_dict[Constants.GaugeType.clock]
        / "clock_model.pt",
        Constants.GaugeType.level: model_folder_path_dict[Constants.GaugeType.level]
        / "level_model.pt",
    }

    model_file_use_dict = {
        Constants.GaugeType.digital: {
            "gauge": model_file_path_dict[Constants.GaugeType.digital],
            "number": model_file_path_dict[Constants.GaugeType.number],
        },
    }

    test_image_path = Path("test_image/")

    test_image_path_dict = {
        Constants.GaugeType.digital: Path("test_image/digital"),
        Constants.GaugeType.number: Path("test_image/number"),
        Constants.GaugeType.dial: Path("test_image/dial"),
        Constants.GaugeType.clock: Path("test_image/clock"),
        Constants.GaugeType.level: Path("test_image/level"),
    }

    inference_class_dict = {
        "gauge": 0,
        "display": 1,
        "frame": 2,
    }
