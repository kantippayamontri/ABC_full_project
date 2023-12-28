from utils import Utils, Constants
from pathlib import Path
from train import YOLOModel, YOLODataset
from inference import (
    InferenceConstants,
    InferenceModel,
    Boxes,
    InferenceUtils,
    DigitalBoxes,
    NumberBoxes,
)
from icecream import ic
import cv2
import torch

gauge_use = Constants.GaugeType.digital

#TODO: load the model
model_path = Path("./models/digital/digital_model.pt")
model = InferenceModel(model_path=model_path,img_target_predict=640,conf=0.25)

#TODO: prepare test data
dataset_path = Path(f"./datasets_for_train/{gauge_use.value}/test/")
dataset_img_bb = Utils.match_img_bb_filename(img_filenames_list=Utils.get_filenames_folder(dataset_path / "images"), bb_filenames_list=None,source_folder=dataset_path)

#TODO: inference model
for idx, (img_path, bb_path) in enumerate(dataset_img_bb):
    image = cv2.imread(str(img_path))
    trans = InferenceUtils.albu_resize_pad_zero(
        target_size=(640,640),
    )

    image_tensor = trans(image=image)["image"]
    image_tensor = torch.div(image_tensor, 255.0)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    gauge_display_frames = model.inference_data(input_img=image_tensor)
    ic(gauge_display_frames)

# TODO:STEP 1: create a confusion matrix 
# TODO:STEP 2: create a precision-recall curve
# TODO:STEP 3: calculate the mean average precision