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

    img_bb_predict_root_path = Path("predict_dataset/")
    img_bb_predict_dict = {
        Constants.GaugeType.digital: {
            "image path": img_bb_predict_root_path / "digital" / "train" / "images",
            "label path": img_bb_predict_root_path / "digital" / "train" /  "labels",
            "yaml path": img_bb_predict_root_path / "digital",
            "bb class": ["gauge", "display", "frame"]
        },
        Constants.GaugeType.number: {
            "image path": img_bb_predict_root_path / "number" / "train" / "images",
            "label path": img_bb_predict_root_path / "number" / "train" / "labels",
        },
    }

    inference_class_dict = {
        "gauge": 0,
        "display": 1,
        "frame": 2,
    }

    inference_number_convert = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: ":",
        11: ".",
        12: "",  # float
        13: "-",
        14: "/",
    }

    predict_parameters = {
        Constants.GaugeType.digital: {
            "image size": (640, 640),
            "conf": 0.3,  # 0.342
        },
        Constants.GaugeType.number: {
            "image size": (640, 640),
            "conf": 0.2,
        },
    }
