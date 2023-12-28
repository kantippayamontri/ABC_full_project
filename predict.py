# FIXME: Inference and Predict share the same Inference module
# TODO: this file use to predict the model -> for each model

from utils import Utils, Constants
from inference import (
    InferenceConstants,
    InferenceModel,
    Boxes,
    InferenceUtils,
    DigitalBoxes,
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
save_img_bb = True
save_result = True

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
if not Utils.check_folder_exists(InferenceConstants.model_file_path_dict[gauge_use]):
    print(
        f"---> [X] model {str(InferenceConstants.model_file_path_dict[gauge_use])} not found ---"
    )
    can_inference = False
else:
    print(
        f"---> [/] model {str(InferenceConstants.model_file_path_dict[gauge_use])} found ---"
    )

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


# TODO: predict load model

model_gauge = InferenceModel(
    model_path=InferenceConstants.model_file_path_dict[gauge_use],
    img_target_predict=InferenceConstants.predict_parameters[
        Constants.GaugeType.digital
    ]["image size"],
    conf=InferenceConstants.predict_parameters[Constants.GaugeType.digital]["conf"],
)

number_gauge = InferenceModel(
    model_path=InferenceConstants.model_file_path_dict[gauge_use],
    img_target_predict=InferenceConstants.predict_parameters[
        Constants.GaugeType.number
    ]["image size"],
    conf=InferenceConstants.predict_parameters[Constants.GaugeType.number]["conf"],
)

# TODO: check folder to save image and bb
if Utils.delete_folder_mkdir(
    folder_path=InferenceConstants.img_bb_predict_root_path, remove=True
):
    if Utils.delete_folder_mkdir(
        folder_path=InferenceConstants.img_bb_predict_dict[gauge_use]["image path"], remove=True
    ):
        print(
            f"--- image path: {InferenceConstants.img_bb_predict_dict[gauge_use]['image path']} has deleted and create ---"
        )
    if Utils.delete_folder_mkdir(
        folder_path=InferenceConstants.img_bb_predict_dict[gauge_use]["label path"], remove=True
    ):
        print(
            f"--- image path: {InferenceConstants.img_bb_predict_dict[gauge_use]['label path']} has deleted and create ---"
        )

start_time = time.time()
for index, input_filename in enumerate(
    Utils.get_filenames_folder(
        source_folder=InferenceConstants.test_image_path_dict[gauge_use]
    )
):
    ic(f"image index: {index}")
    gauge_display_frame_shape = InferenceConstants.predict_parameters[gauge_use][
        "image size"
    ]
    image = cv2.imread(str(input_filename))
    trans = InferenceUtils.albu_resize_pad_zero(
        target_size=gauge_display_frame_shape,
    )

    image_tensor = trans(image=image)["image"]
    image_tensor = torch.div(image_tensor, 255.0)

    image_tensor = torch.unsqueeze(image_tensor, 0)
    output_boxes_model = model_gauge.inference_data(input_img=image_tensor)

    # output_boxes_model = number_gauge.inference_data(input_img=image_tensor)
    image_input_np = torch.squeeze(image_tensor, 0).numpy().transpose(1,2,0)[:,:, [2,1,0]] # convert bgr to rgb format

    if gauge_use == Constants.GaugeType.digital:
        for output_bb in output_boxes_model:
            output_bb_boxes_numpy = output_bb.boxes.cpu().numpy()
            # ic(output_bb_boxes_numpy)
            boxes = DigitalBoxes(
                boxes=output_bb_boxes_numpy.boxes,
                cls=output_bb_boxes_numpy.cls,
                conf=output_bb_boxes_numpy.conf,
                data=output_bb_boxes_numpy.data,
                id=output_bb_boxes_numpy.id,
                is_track=output_bb_boxes_numpy.is_track,
                orig_shape=image_input_np.shape,
                shape=output_bb_boxes_numpy.shape,
                xywh=output_bb_boxes_numpy.xywh,
                xywhn=output_bb_boxes_numpy.xywhn,
                xyxy=output_bb_boxes_numpy.xyxy,
                xyxyn=output_bb_boxes_numpy.xyxyn,
            )
            # ic(boxes.makeBBForSave())

    if save_img_bb:
        ic(f"Process save image and bounding box")

    if save_result:
        ic(f"Process save result")

    # Utils.visualize_img_bb(
    #     img=image_input_np,
    #     bb=boxes.makeBBForSave(),
    #     with_class=True,
    #     labels=["gauge", "display", "frame"],
    #     format=None,
    # )

    Utils.save_image_bb_separate_folder(
        image_np=image_input_np,
        bb_list=boxes.makeBBForSave(),
        image_path=InferenceConstants.img_bb_predict_dict[gauge_use]["image path"] / input_filename.name,
        bb_path=InferenceConstants.img_bb_predict_dict[gauge_use]["label path"] / f"{str(input_filename.stem)}.txt",
        bb_class=InferenceConstants.img_bb_predict_dict[gauge_use]["bb class"],
        yaml_path=InferenceConstants.img_bb_predict_dict[gauge_use]["yaml path"] / f"data.yaml",
    )

end_time = time.time()
number_of_img = len(
    Utils.get_filenames_folder(
        source_folder=InferenceConstants.test_image_path_dict[gauge_use]
    )
)
print(f"number of image: {number_of_img}")
print(
    f"average time: {(end_time-start_time) / len(Utils.get_filenames_folder( source_folder=InferenceConstants.test_image_path_dict[gauge_use]))}"
)

# if index == 0:
#     break
