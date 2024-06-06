from utils import Constants
from ultralytics import YOLO
from train import YOLODataset
from train.models.train_parameters import TrainParameters
import typing
import comet_ml
from train.train_constants import Comet
from icecream import ic
from pathlib import Path
from train.train_constants import TrainModel

class YOLOModel(TrainModel):

    def __init__(
        self,
        model_dict=None,
        use_comet=False,
        gauge_type= None,
        pretrain_path = None,
        version=8,
    ):
        self.model_dict = model_dict
        self.model = self.loadModel(self.mapModel())
        self.use_comet = False # FIX ME: comet
        self.gauge_type = gauge_type
        self.pretrain_path = pretrain_path
        self.version=version
    
    def mapModel(self,):
        if self.model_dict == None:
            return None
        
        model_name = self.model_dict["MODEL"]["MODEL_NAME"]
        model_type = self.model_dict["MODEL"]["MODEL_TYPE"]
        model_version = self.model_dict["MODEL"]["MODEL_VERSION"]

        # for pretrain model
        model_path = self.model_dict["MODEL_PATH"]

        # dict for store model value from () keys
        map_model = {
            # yolo version 8
            (8,"n" ): "yolov8n.pt",
            (8,"s"): "yolov8s.pt",
            (8,"m"): "yolov8m.pt",
            (8,"l"): "yolov8l.pt",
            (8,"x"): "yolov8x.pt",
            # yolo version 0
            # (9,"t"):"yolov9t.pt",
            # (9,"s"):"yolov9s.pt",
            # (9,"m"):"yolov9m.pt",
            (9,"c"):"yolov9c.pt",
            (9,"e"):"yolov9e.pt",
            # yolo version 10
            # (10,"n"):"yolov10n.pt",
        }
        
        ic(model_name, model_type, model_version, model_path)
        
        return map_model[(model_version, model_type)]

    
    
    def loadModel(self, model_path: Path):
        _model = None
        print(f"--- Load Model :{str(model_path)} ---")
        _model = YOLO(str(model_path)) # load a model (.pt = with pretrain, .yaml = without pretrain)
        print(f"--- Load Model Success ---")
        return _model


    def trainModel(self, train_parameters: TrainParameters):
        # make experiments in comet ml
        if self.use_comet:
            self.init_comet(train_parameters=train_parameters)

        # train model 
        self.model.train(
            data=str(train_parameters.get_data_yaml_path()),
            epochs=train_parameters.get_epochs(),
            imgsz=train_parameters.get_imgsz(),
            batch=train_parameters.get_batch(),
            cache=train_parameters.get_cache(),
            patience=train_parameters.get_patience(),
            device=train_parameters.get_device(),
            workers=train_parameters.get_workers(),
            resume=train_parameters.get_resume(),
            lr0=train_parameters.get_learning_rate(),
            lrf=train_parameters.get_final_learning_rate(),
            project=str(train_parameters.get_project_name()), #TODO: path for experimental_project folder
            name=train_parameters.get_name()[0], # TODO: name of the experiment
            fliplr=0.0, #set flip left and right to zero
        )
        
        
        # end the experiment in comet ml
        if self.use_comet:
            self.end()

        
    def init_comet(
        self,
        train_parameters:TrainParameters=None
    ):
        comet_ml.init()

        comet_parameter = Comet.parameters[self.gauge_type]

        ic(comet_parameter)

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

        
        return None
        # return super().trainModel(train_parameters)

    
    def exportModel(self, dest_path: Path):
        return super().exportModel(dest_path)
    
    def evalModel(self):
        return super().evalModel()

# class YOLOModel:
#     def __init__(
#         self,
#         model_type,
#         use_comet=False,
#         gauge_type= None,
#         pretrain_path = None,
#         version=8,
#     ):
#         self.model_type = model_type
#         self.model = self.loadModels(model_type)
#         self.use_comet = use_comet
#         self.gauge_type = gauge_type
#         self.pretrain_path = pretrain_path
#         self.version=version

#     def loadModels(self, model_type):
#         model = None
#         if model_type == Constants.ModelType.NANO:
#             print(f"\t--- MODEL NANO ---")
#             model = YOLO("yolov8n.yaml")  # build a new model from scratch
#             model = YOLO(
#                 "yolov8n.pt"
#             )  # load a pretrained model (recommended for training)
#             model = YOLO("yolov8n.yaml").load("yolov8n.pt")
#             print(f"\t--- YOLOv8 NANO Load Success ---")
#         elif model_type == Constants.ModelType.SMALL:
#             print(f"\t--- MODEL small ---")
#             model = YOLO("yolov8s.yaml")  # build a new model from scratch
#             model = YOLO(
#                 "yolov8s.pt"
#             )  # load a pretrained model (recommended for training)
#             model = YOLO("yolov8s.yaml").load("yolov8s.pt")
#             print(f"\t--- YOLOv8 SMALL Load Success ---")
#         elif model_type == Constants.ModelType.MEDIUM:
#             print(f"\t--- MODEL MEDIUM ---")
#             # model = YOLO("yolov8m.yaml")  # build a new model from scratch
#             model = YOLO(
#                 "yolov8m.pt"
#             )  # load a pretrained model (recommended for training)
#             model = YOLO("yolov8m.yaml").load("yolov8m.pt")
#             print(f"\t--- YOLOv8 MEDIUM Load Success ---")
#         elif model_type == Constants.ModelType.LARGE:
#             print(f"\t--- MODEL LARGE ---")
#             model = YOLO("yolov8l.yaml")  # build a new model from scratch
#             model = YOLO(
#                 "yolov8l.pt"
#             )  # load a pretrained model (recommended for training)
#             model = YOLO("yolov8l.yaml").load("yolov8l.pt")
#             print(f"\t--- YOLOv8 LARGE Load Success ---")
#         elif model_type == Constants.ModelType.EXTRA_LARGE:
#             print(f"\t--- MODEL EXTRA LARGE ---")
#             model = YOLO("yolov8x.yaml")  # build a new model from scratch
#             model = YOLO(
#                 "yolov8x.pt"
#             )  # load a pretrained model (recommended for training)
#             model = YOLO("yolov8x.yaml").load("yolov8x.pt")
#             print(f"\t--- YOLOv8 EXTRA LARGE Load Success ---")
#         return model

#     def train(self, parameters: TrainParameters):
#         # if self.use_comet:
#         #     self.init_comet(train_parameters=parameters)
            
#         ic(parameters.comet_parameters())
#         # TODO: check experiment folder
        
#         # TODO: create name of the experiment
        
#         ic(str(parameters.get_project_name()[0]))
#         ic(parameters.get_project_name())
#         ic(parameters.get_name())
#         ic(str(parameters.get_data_yaml_path()))
        
#         self.model.train(
#             data=str(parameters.get_data_yaml_path()),
#             epochs=parameters.get_epochs(),
#             imgsz=parameters.get_imgsz(),
#             batch=parameters.get_batch(),
#             cache=parameters.get_cache(),
#             patience=parameters.get_patience(),
#             device=parameters.get_device(),
#             workers=parameters.get_workers(),
#             resume=parameters.get_resume(),
#             lr0=parameters.get_learning_rate(),
#             lrf=parameters.get_final_learning_rate(),
#             project=str(parameters.get_project_name()), #TODO: path for experimental_project folder
#             name=parameters.get_name()[0], # TODO: name of the experiment
#             fliplr=0.0, #set flip left and right to zero
#         )
        
#         # if self.use_comet:
#         #     self.end_comet()

#     def init_comet(
#         self,
#         train_parameters:TrainParameters=None
#     ):
#         comet_ml.init()

#         comet_parameter = Comet.parameters[self.gauge_type]

#         ic(comet_parameter)

#         # Initialize Comet ML with API key
#         experiment = comet_ml.Experiment(
#             api_key=comet_parameter["api_key"],
#             project_name=comet_parameter["project_name"],
#             workspace=comet_parameter["workspace"],
#         )
        
#         # Log parameters
#         experiment.log_parameters(parameters=train_parameters.comet_parameters())
    
#     def end_comet(self,):
#         comet_ml.Experiment.end() # end the experiment
#         return
    
#     def export_model(self, ):
#         self.model
#         return
    