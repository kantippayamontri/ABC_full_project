from utils import Constants
from ultralytics import YOLO
from train import YOLODataset
from train.models.train_parameters import TrainParameters
import typing
import comet_ml
from train.train_constants import Comet
from icecream import ic
from pathlib import Path


class YOLOModel:
    def __init__(
        self,
        model_type,
        use_comet=False,
        gauge_type: Constants.GaugeType = None,
    ):
        self.model_type = model_type
        self.model = self.loadModels(model_type)
        self.use_comet = use_comet
        self.gauge_type = gauge_type

    def loadModels(self, model_type):
        model = None
        if model_type == Constants.ModelType.NANO:
            print(f"\t--- MODEL NANO ---")
            model = YOLO("yolov8n.yaml")  # build a new model from scratch
            model = YOLO(
                "yolov8n.pt"
            )  # load a pretrained model (recommended for training)
            model = YOLO("yolov8n.yaml").load("yolov8n.pt")
            print(f"\t--- YOLOv8 Load Success ---")
        elif model_type == Constants.ModelType.SMALL:
            print(f"\t--- MODEL small ---")
            model = YOLO("yolov8s.yaml")  # build a new model from scratch
            model = YOLO(
                "yolov8s.pt"
            )  # load a pretrained model (recommended for training)
            model = YOLO("yolov8s.yaml").load("yolov8s.pt")
            print(f"\t--- YOLOv8 Load Success ---")
        elif model_type == Constants.ModelType.MEDIUM:
            print(f"\t--- MODEL MEDIUM ---")
            model = YOLO("yolov8m.yaml")  # build a new model from scratch
            model = YOLO(
                "yolov8m.pt"
            )  # load a pretrained model (recommended for training)
            model = YOLO("yolov8m.yaml").load("yolov8m.pt")
            print(f"\t--- YOLOv8 Load Success ---")
        elif model_type == Constants.ModelType.LARGE:
            print(f"\t--- MODEL LARGE ---")
            model = YOLO("yolov8l.yaml")  # build a new model from scratch
            model = YOLO(
                "yolov8l.pt"
            )  # load a pretrained model (recommended for training)
            model = YOLO("yolov8l.yaml").load("yolov8l.pt")
            print(f"\t--- YOLOv8 Load Success ---")
        elif model_type == Constants.ModelType.EXTRA_LARGE:
            print(f"\t--- MODEL EXTRA LARGE ---")
            model = YOLO("yolov8x.yaml")  # build a new model from scratch
            model = YOLO(
                "yolov8x.pt"
            )  # load a pretrained model (recommended for training)
            model = YOLO("yolov8x.yaml").load("yolov8x.pt")
            print(f"\t--- YOLOv8 Load Success ---")
        return model

    def train(self, parameters: TrainParameters):
        if self.use_comet:
            self.init_comet(train_parameters=parameters)
            
        print(f"train parameters: {parameters.comet_parameters()}")
        # TODO: check experiment folder
        
        # TODO: create name of the experiment
        
        ic(str(parameters.get_project_name()[0]))
        
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
            lr0=parameters.get_learning_rate(),
            lrf=parameters.get_final_learning_rate(),
            project=f"{str(parameters.get_project_name()[0])}", #TODO: path for experimental_project folder
            name=f"{str(parameters.get_name()[0])}" # TODO: name of the experiment
        )
        
        if self.use_comet:
            self.end_comet()

    def init_comet(
        self,
        train_parameters:TrainParameters=None
    ):
        comet_ml.init()

        comet_parameter = Comet.parameters[self.gauge_type.value]

        # ic(comet_parameter)

        # Initialize Comet ML with API key
        experiment = comet_ml.Experiment(
            api_key=comet_parameter["api_key"],
            project_name=comet_parameter["project_name"],
            workspace=comet_parameter["workspace"],
        )
        
        # Log parameters
        experiment.log_parameters(parameters=train_parameters.comet_parameters())
    
    def end_comet(self,):
        comet_ml.Experiment.end() # end the experiment
        return
        
        

    # def set_up_comet_ml(
    #     self,
    # ):
    #     if self.use_comet:
    #         if self.dataset_type is not None and self.dataset_use is not None:
    #             print(f"--- use comet ML ---")
    #             experiment_parameters = Comet.parameters[Constants.DatasetType.TYPE_3][Constants.DatasetUse.TYPE_3_GAUGE_DISPLAY_FRAME]
    #             return Experiment(
    #                 api_key=experiment_parameters['api_key'],
    #                 project_name=experiment_parameters['project_name'],
    #                 workspace=experiment_parameters['workspace'],
    #             )
    #         else:
    #             print(f"--- Can not use Comet ML cuz you need to add dataset_type or dataset_use parameters---")
    #     else:
    #         print(f"--- Don't use comet ML ---")
    #         return None

    #     return None
