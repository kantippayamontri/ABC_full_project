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

    # def trainModel(self, parameters: TrainParameters):
    #     # if self.use_comet:
    #     #     self.init_comet(train_parameters=parameters)
            
    #     ic(parameters.comet_parameters())
    #     # TODO: check experiment folder
        
    #     # TODO: create name of the experiment
        
    #     ic(str(parameters.get_project_name()[0]))
    #     ic(parameters.get_project_name())
    #     ic(parameters.get_name())
        
    #     self.model.train(
    #         data=parameters.get_data_yaml_path(),
    #         epochs=parameters.get_epochs(),
    #         imgsz=parameters.get_imgsz(),
    #         batch=parameters.get_batch(),
    #         cache=parameters.get_cache(),
    #         patience=parameters.get_patience(),
    #         device=parameters.get_device(),
    #         workers=parameters.get_workers(),
    #         resume=parameters.get_resume(),
    #         lr0=parameters.get_learning_rate(),
    #         lrf=parameters.get_final_learning_rate(),
    #         project=parameters.get_project_name(), #TODO: path for experimental_project folder
    #         name=parameters.get_name()[0], # TODO: name of the experiment
    #         fliplr=0.0, #set flip left and right to zero
    #     )
        
    #     # if self.use_comet:
    #     #     self.end_comet()

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
    
    def export_model(self, ):
        self.model
        return

    def loadModel(self,model_path):
        return YOLO(str(model_path))

    def evalModel(self):
        return 
    
    def exportModel(self, dest_path: Path):
        return super().exportModel(dest_path)
    
    def trainModel(self, train_parameters: TrainParameters, **kwargs):
        train_with_command = True
        if train_with_command:
            self.trainModelCommand(train_parameters=train_parameters, model_path=kwargs["model_path"])
        else:
            self.trainModelCode(parameters=train_parameters)
        
        return 
    
    def trainModelCode(self, parameters: TrainParameters):
        # if self.use_comet:
        #     self.init_comet(train_parameters=parameters)
            
        ic(parameters.comet_parameters())
        # TODO: check experiment folder
        
        # TODO: create name of the experiment
        
        ic(str(parameters.get_project_name()[0]))
        ic(parameters.get_project_name())
        ic(parameters.get_name())

        print(f"+++=== Train with Code ===+++")
        
        self.model.train(
            data=parameters.get_data_yaml_path(),
            epochs=parameters.get_epochs(),
            imgsz=parameters.get_imgsz(),
            batch=parameters.get_batch(),
            # cache=parameters.get_cache(),
            patience=parameters.get_patience(),
            device=parameters.get_device(),
            workers=parameters.get_workers(),
            resume=parameters.get_resume(),
            lr0=parameters.get_learning_rate(),
            lrf=parameters.get_final_learning_rate(),
            project=parameters.get_project_name(), #TODO: path for experimental_project folder
            name=parameters.get_name()[0], # TODO: name of the experiment
            fliplr=0.0, #set flip left and right to zero
            optimizer="Adam",
        )
        
        # if self.use_comet:
        #     self.end_comet()

    def trainModelCommand(self, train_parameters: TrainParameters, model_path: Path):
        from icecream import ic

        print(f"+++=== Train with Command ===+++")
        
        # train the model
        command = "yolo detect train "
        command += f"model={str(model_path)} "
        command += f"data={str(train_parameters.get_data_yaml_path())} "
        command += f"epochs={train_parameters.get_epochs()} "
        command += f"imgsz={train_parameters.get_imgsz()} "
        command += f"batch={train_parameters.get_batch()} "
        command += f"cache={train_parameters.get_cache()} " 
        command += f"patience={train_parameters.get_patience()} "
        command += f"device={train_parameters.get_device()} "
        command += f"workers={train_parameters.get_workers()} "

        # bug we need to avoid resume for now
        # if train_parameters.get_resume():
        #     command += f"resume "

        command += f"lr0={train_parameters.get_learning_rate()} "
        command += f"lrf={train_parameters.get_final_learning_rate()} "
        command += f"project='{str(train_parameters.get_project_name())}' "
        command += f"name='{train_parameters.get_name()[0]}' "
        command += f"fliplr=0.0 " # set flip left and right to zero
        command += f"optimizer=Adam mosaic=0.5 scale=0.1" 
        # command += f""
        # command += f""

        # train the model
        from subprocess import call
        ic(f"command: {command}")
        print(command)
        call(command, shell=True)

        return 
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       