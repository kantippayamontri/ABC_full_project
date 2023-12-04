# TODO: this file use to predict the model
from utils import Utils, Constants
from inference import (
    InferenceConstants,
    InferenceModel,
    Boxes,
    InferenceUtils,
    DigitalBoxes,
    NumberBoxes,
)
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image
from icecream import ic
import cv2
from PIL import Image
import time

gauge_use = Constants.GaugeType.digital

# TODO: step 1: check model folder

# check all model folders
for key, folder in InferenceConstants.model_folder_path_dict.items():
    if not Utils.check_folder_exists(folder_path=folder):
        print(f"--->[X] not found {folder}")
        Utils.delete_folder_mkdir(folder_path=folder, remove=False)
    else:
        print(f"--->[/] found {folder}")
        if Utils.check_folder_exists(
            folder_path=InferenceConstants.model_file_path_dict[key]
        ):
            print(f"\t---> [/] found model")
        else:
            print(f"\t---> [X] not found model")
print()
print(f"Model use for {gauge_use.value}")
# check model folder path

can_inference = True
for model_name, model_path in InferenceConstants.model_file_use_dict[gauge_use].items():
    if not Utils.check_folder_exists(model_path):
        print(f"---> [X] model {str(model_path)} not found ---")
        can_inference = False
    else:
        print(f"---> [/] model {str(model_path)} found ---")

if not can_inference:
    exit()

# TODO: step 2: check test image folder
print()
print(f"Checking test image")
if not Utils.check_folder_exists(InferenceConstants.test_image_path):
    Utils.delete_folder_mkdir(
        folder_path=InferenceConstants.test_image_path, remove=False
    )
    print(f"--- can not find test image folder ---")
    exit()

for key, value in InferenceConstants.test_image_path_dict.items():
    if Utils.check_folder_exists(folder_path=value):
        print(
            f"---> [/] found {str(value)}, number of files is {Utils.count_files(value)} images. ---"
        )
    else:
        print(f"---> [X] not found {str(value)} ---")
        Utils.delete_folder_mkdir(folder_path=value, remove=False)
# check number of files in test image folder
if Utils.count_files(InferenceConstants.test_image_path_dict[gauge_use]) == 0:
    print(
        f"--- pls add test image to {str(InferenceConstants.test_image_path_dict[gauge_use])}"
    )
    exit()

if gauge_use == Constants.GaugeType.digital:
    model_gauge = InferenceModel(
        model_path=InferenceConstants.model_file_use_dict[gauge_use]["gauge"],
        img_target_predict= InferenceConstants.predict_parameters[Constants.GaugeType.digital]["image size"],
        conf=InferenceConstants.predict_parameters[Constants.GaugeType.digital]["conf"]

    )
    number_gauge = InferenceModel(
        model_path=InferenceConstants.model_file_use_dict[gauge_use]["number"],
        img_target_predict=InferenceConstants.predict_parameters[Constants.GaugeType.number]["image size"],
        conf=InferenceConstants.predict_parameters[Constants.GaugeType.number]["conf"]
    )
    start_time = time.time()
    number_samples = len(
        Utils.get_filenames_folder(
            source_folder=InferenceConstants.test_image_path_dict[gauge_use]
        )
    )
    for index, input_filename in enumerate(
        Utils.get_filenames_folder(
            source_folder=InferenceConstants.test_image_path_dict[gauge_use]
        )
    ):
        gauge_display_frame_shape = InferenceConstants.predict_parameters[
            Constants.GaugeType.digital
        ]["image size"]
        image = cv2.imread(str(input_filename))
        trans = InferenceUtils.albu_resize_pad_zero(
            target_size=gauge_display_frame_shape,
        )

        image_tensor = trans(image=image)["image"]
        image_tensor = torch.div(image_tensor, 255.0)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        gauge_display_frames = model_gauge.inference_data(input_img=image_tensor)
        ic(f"image index: {index}")
        for gauge_display_frame in gauge_display_frames:
            gdf = gauge_display_frame.boxes.cpu().numpy()
            boxes = DigitalBoxes(
                boxes=gdf.boxes,
                cls=gdf.cls,
                conf=gdf.conf,
                data=gdf.data,
                id=gdf.id,
                is_track=gdf.is_track,
                orig_shape=gdf.orig_shape,
                shape=gdf.shape,
                xywh=gdf.xywh,
                xywhn=gdf.xywhn,
                xyxy=gdf.xyxy,
                xyxyn=gdf.xyxyn,
            )

        ic(f"number gauge: {boxes.nGauges}")
        ic(f"number display: {boxes.nDisplays}")
        ic(f"number frame: {boxes.nFrames}")
        ic(f"number frame in gauge: {len(boxes.frameInGauge)}")
        ic(f"number frame in display: {len(boxes.frameInDisplay)}")

        # ic(f"loop frame")
        # loop frame images
        ic(f"image index: {index}")
        for frame_index, frame in enumerate(boxes.makeFrameForPredict()):
            ic(f"\tframe index: {frame_index}")
            frame_shape = InferenceConstants.predict_parameters[
                Constants.GaugeType.number
            ]["image size"]
            # get position in real image
            ori_frame_coor = boxes.getCoordinatesRealImage(
                ori_shape=gauge_display_frame_shape,
                want_shape=image.shape,
                box_coor=frame.xyxy,
            )
            frame_transform = InferenceUtils.albu_make_frame(
                img=image, frame_coor=ori_frame_coor, target_size=frame_shape
            )
            crop_frame_image = frame_transform(image=image)["image"]  # tensor
            crop_frame_image = torch.div(crop_frame_image, 255.0)
            crop_frame_image = torch.unsqueeze(crop_frame_image, 0)

            number_predicts = number_gauge.inference_data(input_img=crop_frame_image)
            for number_predict in number_predicts:
                # ic(number_predict)
                nbp = number_predict.boxes.cpu().numpy()
                boxes = NumberBoxes(
                    boxes=nbp.boxes,
                    cls=nbp.cls,
                    conf=nbp.conf,
                    data=nbp.data,
                    id=nbp.id,
                    is_track=nbp.is_track,
                    orig_shape=nbp.orig_shape,
                    shape=nbp.shape,
                    xywh=nbp.xywh,
                    xywhn=nbp.xywhn,
                    xyxy=nbp.xyxy,
                    xyxyn=nbp.xyxyn,
                )
                # [ box.printBox()for box in boxes.boxes_list]
                ic(boxes.predict_number())
        Utils.visualize_img_bb(
            img=torch.squeeze(image_tensor, 0).numpy().transpose(1, 2, 0),
            bb=[],
            with_class=False,
            labels=None,
            format=None,
        )

        # predict number

        # print(torch.squeeze(image_tensor, 0).numpy().transpose(1, 2, 0).shape)

        if index == 2:
            break
        # if index == 3:
        #     break
    end_time = time.time()
    ic(f"use time : {(end_time - start_time ) / number_samples}")
