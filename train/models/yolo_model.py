from utils import Constants
from ultralytics import YOLO
from train import YOLODataset
from train.models.train_parameters import TrainParameters
import typing
from comet_ml import Experiment
from train.train_constants import Comet

class YOLOModel:
    def __init__(
        self,
        model_type,
        use_comet=False,
        dataset_type: Constants.DatasetType = None,
        dataset_use: Constants.DatasetUse = None,
    ):
        self.model_type = model_type
        self.model = self.loadModels(model_type)
        self.set_up_comet_ml()
        self.use_comet = use_comet #comet
        self.dataset_type = dataset_type #comet
        self.dataset_use = dataset_use #comet
        self.experiment = self.set_up_comet_ml() #comet

    def loadModels(self, model_type):
        model = None
        if model_type == Constants.ModelType.NANO:
            print(f"--- MODEL NANO ---")
            model = YOLO("yolov8n.yaml")  # build a new model from scratch
            model = YOLO(
                "yolov8n.pt"
            )  # load a pretrained model (recommended for training)
            model = YOLO("yolov8n.yaml").load("yolov8n.pt")
            print(f"--- YOLOv8 Load Success ---")
        elif model_type == Constants.ModelType.SMALL:
            print(f"--- MODEL small ---")
            model = YOLO("yolov8s.yaml")  # build a new model from scratch
            model = YOLO(
                "yolov8s.pt"
            )  # load a pretrained model (recommended for training)
            model = YOLO("yolov8s.yaml").load("yolov8s.pt")
            print(f"--- YOLOv8 Load Success ---")
        elif model_type == Constants.ModelType.MEDIUM:
            print(f"--- MODEL MEDIUM ---")
            model = YOLO("yolov8m.yaml")  # build a new model from scratch
            model = YOLO(
                "yolov8m.pt"
            )  # load a pretrained model (recommended for training)
            model = YOLO("yolov8m.yaml").load("yolov8m.pt")
            print(f"--- YOLOv8 Load Success ---")
        elif model_type == Constants.ModelType.LARGE:
            print(f"--- MODEL LARGE ---")
            model = YOLO("yolov8l.yaml")  # build a new model from scratch
            model = YOLO(
                "yolov8l.pt"
            )  # load a pretrained model (recommended for training)
            model = YOLO("yolov8l.yaml").load("yolov8l.pt")
            print(f"--- YOLOv8 Load Success ---")
        elif model_type == Constants.ModelType.EXTRA_LARGE:
            print(f"--- MODEL EXTRA LARGE ---")
            model = YOLO("yolov8x.yaml")  # build a new model from scratch
            model = YOLO(
                "yolov8x.pt"
            )  # load a pretrained model (recommended for training)
            model = YOLO("yolov8x.yaml").load("yolov8x.pt")
            print(f"--- YOLOv8 Load Success ---")
        return model

    def train(self, parameters: TrainParameters):
        self.model.train(
            data=parameters.get_data_yaml_path(),
            epochs=parameters.get_epochs(),
            imgsz=parameters.get_imgsz(),
            batch=parameters.get_batch(),
            cache=parameters.get_cache(),
            patience=parameters.get_patience(),
            device=parameters.get_device(),
            workers=parameters.get_workers(),
            resume=parameters.get_resume(),
        )
        

    def set_up_comet_ml(
        self,
    ):
        if self.use_comet:
            if self.dataset_type is not None and self.dataset_use is not None:
                print(f"--- use comet ML ---")
                experiment_parameters = Comet.parameters[Constants.DatasetType.TYPE_3][Constants.DatasetUse.TYPE_3_GAUGE_DISPLAY_FRAME]
                return Experiment(
                    api_key=experiment_parameters['api_key'],
                    project_name=experiment_parameters['project_name'],
                    workspace=experiment_parameters['workspace'],
                )
            else:
                print(f"--- Can not use Comet ML cuz you need to add dataset_type or dataset_use parameters---")
        else:
            print(f"--- Don't use comet ML ---")
            return None

        return None
