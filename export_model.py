# TODO: this use for export the model
from pathlib import Path
from utils import Constants
from utils import Utils
from icecream import ic
from ultralytics import YOLO

model_root_path = Constants.model_experiment_path
model_path = model_root_path / "train7"
model_weights_path = model_path / "weights" / "best.pt"
model_arguments_path = model_path / "args.yaml"

# TODO: load the model
model_arguments_yaml = Utils.read_yaml_file(yaml_file_path=model_arguments_path)
ic(model_arguments_yaml)
model = YOLO(model_arguments_yaml['model'])
model = YOLO(str(model_weights_path))

# TODO: val the model


# TODO:  export for torchscript (for mobile)
model.export(format="torchscript", imgsz=model_arguments_yaml['imgsz'], optimize=True)