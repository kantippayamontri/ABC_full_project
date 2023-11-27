from ultralytics import YOLO
import torch
from icecream import ic


class InferenceModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model(self.model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path):
        model = YOLO(model_path)
        return model

    def inference_data(self, input_img):
        input_img = input_img.to(self.device) 
        # ic(f"input image shape: {input_img.shape}")
        output = self.model.predict(input_img, imgsz=1024, conf=0.5,device=self.device)

        return output