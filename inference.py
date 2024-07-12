# TODO: this file use to predict the model
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision.transforms as transforms
from icecream import ic
from PIL import Image
from ultralytics import YOLO

from inference import (
    Boxes,
    ClockBoxes,
    DigitalBoxes,
    InferenceClock,
    InferenceConstants,
    InferenceModel,
    InferenceUtils,
    NumberBoxes,
)
from utils import Constants, Utils

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="for training arguments parser")

# Add positional argument
parser.add_argument("--input_file", type=str, help="your python file to run")
parser.add_argument("--gauge_use", type=str, help="type gauge")
parser.add_argument("--img_path", type=str, help="image path")

# # Add optional argument with a default value
parser.add_argument(
    "--select_frame",
    type=int,
    default=0,
    help="selcet_frame from path and move to select_frame -> work only in number gauge type",
)

# Parse the command-line arguments
args = parser.parse_args()

gauge_use = Utils.get_enum_by_value(
    value=args.gauge_use.lower(), enum=Constants.GaugeType
)
img_path = args.img_path

image_size_dict = {
    Constants.GaugeType.digital: (640, 640),
    Constants.GaugeType.number: (640, 640),
    Constants.GaugeType.clock: (640, 640),
}
model_path_dict = {
    Constants.GaugeType.digital: {
        "digital": Path(""),
        "number": Path("./models/number/best (2).pt"),
    },
    Constants.GaugeType.number: Path(""),
    Constants.GaugeType.clock: {
        "clock gauge": Path(
            "/home/kansmarts777/wsl_code/ABC_full_project/clock_gauge_color.pt"
        ),
        "clock inside": "/home/kansmarts777/wsl_code/ABC_full_project/clock_inside_gray.pt",
    },
}

if gauge_use == Constants.GaugeType.digital:
    model_gauge = InferenceModel(
        model_path=model_path_dict[gauge_use]["digital"],
        img_target_size=image_size_dict[gauge_use],
        conf=0.25,
    )
    number_gauge = InferenceModel(
        model_path=model_path_dict[gauge_use]["number"],
        img_target_size=image_size_dict[Constants.GaugeType.number],
        conf=0.25,
    )

    number_samples = len(
        Utils.get_filenames_folder(
            source_folder=Path(img_path),
        )
    )

    start_time = time.time()
    for digital_index, digital_filename in enumerate(
        Utils.get_filenames_folder(source_folder=Path(img_path))
    ):
        # if digital_index != 1:
        #     continue
        ic(f"digital index: {digital_index}, filename: {digital_filename}")
        image = cv2.imread(str(digital_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        trans = InferenceUtils.albu_resize_pad_zero(
            target_size=image_size_dict[gauge_use],
            gray=True,
        )
        trans_rgb = InferenceUtils.albu_resize_pad_zero(
            target_size=image_size_dict[gauge_use],
            gray=False,
        )  # not convert to gray scale image

        image_tensor = trans(image=image, bboxes=[])["image"]
        image_tensor = torch.div(image_tensor, 255.0)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        image_tensor_rgb = trans_rgb(image=image, bboxes=[])["image"]
        image_tensor_rgb = torch.div(image_tensor_rgb, 255.0)
        image_tensor_rgb = torch.unsqueeze(image_tensor_rgb, 0)

        gauge_display_frames = model_gauge.inference_data(input_img=image_tensor)

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

        # ic(f"number gauge: {boxes.nGauges}")
        # ic(f"number display: {boxes.nDisplays}")
        # ic(f"number frame: {boxes.nFrames}")
        # ic(f"number frame in gauge: {len(boxes.frameInGauge)}")
        # ic(f"number frame in display: {len(boxes.frameInDisplay)}")
        # ic(boxes.boxes)
        # ic(boxes.makeBBForSave())
        # ic(gauge_display_frames[0].boxes[0].boxes)

        # Utils.visualize_img_bb(
        #     # img=torch.squeeze(image_tensor, 0).numpy().transpose(1, 2, 0),
        #     img=torch.squeeze(image_tensor_rgb, 0).numpy().transpose(1, 2, 0),
        #     bb=[
        #         {"class": bb[5], "bb": [(bb[0], bb[1]), (bb[2], bb[3])]}
        #         for bb in boxes.boxes
        #     ],
        #     with_class=True,
        #     labels=["gauge", "display", "frame"],
        #     format=Constants.BoundingBoxFormat.XYXY,
        # )

        image_tensor_np = torch.squeeze(image_tensor_rgb, 0).numpy().transpose(1, 2, 0)

        for frame_index, frame in enumerate(boxes.framePredict):
            ic(f"\tframe index: {frame_index}")
            frame_shape = image_size_dict[Constants.GaugeType.number]

            frame_transform = InferenceUtils.albu_make_frame(
                img_shape=frame_shape, frame_coor=frame.xyxy, target_size=frame_shape
            )
            crop_frame_image = frame_transform(image=image_tensor_np)["image"]  # tensor
            # crop_frame_image = torch.div(crop_frame_image, 255.0)
            crop_frame_image = torch.unsqueeze(crop_frame_image, 0)

            # Utils.visualize_img_bb(
            #     img=torch.squeeze(crop_frame_image, 0).numpy().transpose(1, 2, 0),
            #     bb=[],
            #     with_class=False,
            #     labels=None,
            #     format=None,
            # )

            number_predicts = number_gauge.inference_data(input_img=crop_frame_image)
            for number_predict in number_predicts:
                nbp = number_predict.boxes.cpu().numpy()
                boxes = NumberBoxes(
                    boxes=nbp.data,
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
                    image=torch.squeeze(crop_frame_image, 0).numpy().transpose(1, 2, 0),
                )
                ic(boxes.predict_number())
            # break #FIXME: remove this

        # if digital_index ==0:
        #     exit()
    end_time = time.time()
    ic(f"use time : {(end_time - start_time ) / number_samples}")

elif gauge_use == Constants.GaugeType.number:
    ic(f"--- Inference Number ---")
    number_model = InferenceModel(
        model_path=model_path_dict[gauge_use],  # only one model
        img_target_size=image_size_dict[gauge_use],
        conf=0.25,
    )

    samples = Utils.get_filenames_folder(
        source_folder=Path(img_path),
    )

    number_samples = len(samples)

    if args.select_frame:
        Utils.delete_folder_mkdir(folder_path=Path("./select_frame"), remove=True)

    start_time = time.time()

    for frame_index, frame_image_path in enumerate(samples):
        ic(f"\tframe index: {frame_index}")

        image = cv2.imread(str(frame_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        trans = InferenceUtils.albu_resize_pad_zero(
            target_size=image_size_dict[gauge_use],
            gray=False,
        )

        image_tensor = trans(image=image, bboxes=[])["image"]
        image_tensor = torch.div(image_tensor, 255.0)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        gauge_display_frames = number_model.inference_data(input_img=image_tensor)

        # Utils.visualize_img_bb(
        #     img=torch.squeeze(image_tensor, 0).numpy().transpose(1, 2, 0),
        #     bb=[],
        #     with_class=False,
        #     labels=None,
        #     format=None,
        # )

        number_predicts = number_model.inference_data(input_img=image_tensor)
        for number_predict in number_predicts:
            nbp = number_predict.boxes.cpu().numpy()
            boxes = NumberBoxes(
                boxes=nbp.data,
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
                image=torch.squeeze(image_tensor, 0).numpy().transpose(1, 2, 0),
            )
            # ic(boxes.predict_number())
            boxes.predict_number()
            if args.select_frame and (boxes.predict_number != ""):
                ic(f"--- select frame ---")
                Utils.copy_file(
                    source_file_path=frame_image_path,
                    target_file_path=Path("./select_frame/"),
                )

    end_time = time.time()
    ic(f"use time : {(end_time - start_time ) / number_samples}")

elif gauge_use == Constants.GaugeType.clock:
    print(f"--- Inference clock ---")

    model_clock = InferenceModel(
        model_path=model_path_dict[gauge_use]["clock gauge"],
        img_target_size=image_size_dict[gauge_use],
        conf=0.25,
    )

    model_clock_inside = InferenceModel(
        model_path=model_path_dict[gauge_use]["clock inside"],
        img_target_size=image_size_dict[gauge_use],
        conf=0.1,
    )

    samples = Utils.get_filenames_folder(source_folder=Path(img_path))

    number_samples = len(samples)

    start_time = time.time()

    for clock_index, clock_img_path in enumerate(samples):
        print(f"clock index: {clock_index}")

        # print(f"image tensor shape: {image_tensor.shape}")

        # TODO: load image
        original_image = cv2.imread(str(clock_img_path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Utils.visualize_img_bb(
        #     img=original_image,
        #     bb=[],
        #     with_class=False,
        #     labels=None,
        #     format=None,
        # )
        # gray_img = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        # gray_img = cv2.cvtColor(
        #     gray_img, cv2.COLOR_GRAY2BGR
        # )  # convert from 1 channel to 3 channel gray scale
        # gray_img = torch.from_numpy(gray_img)
        # gray_img = torch.div(gray_img, 255.0)
        # gray_img = np.transpose(gray_img, (2, 0, 1))

        input_clock_gauge = torch.from_numpy(original_image)
        input_clock_gauge = torch.div(input_clock_gauge,255.0)
        input_clock_gauge = np.transpose(input_clock_gauge, (2,0,1))


        # TODO: find the gauge and crop the image
        clock_gauge_result = model_clock.inference_data(
            input_img=torch.unsqueeze(input_clock_gauge, 0)
        )

        gauge_list = []
        for r in clock_gauge_result:
            nbp = r.boxes.cpu().numpy()

            boxes = ClockBoxes(
                boxes=nbp.data,
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
            gauge_list = boxes.gauge_list

        # ic(f"number of gauge: {len(gauge_list)}")

        # ic(gauge_list[0].xyxy, gauge_list[0].cls)
        Utils.visualize_img_bb(
            img=np.transpose(input_clock_gauge, (1, 2, 0)),
            bb=[
                {"class": 0, "bb": [(g.xyxy[0], g.xyxy[1]), (g.xyxy[2], g.xyxy[3])]}
                for g in gauge_list
            ],
            with_class=True,
            labels=["gauge"],
            format=Constants.BoundingBoxFormat.XYXY,
        )

        if len(gauge_list) == 0:
            print(f"Can not find the gauge")
            break

        # TODO: crop the gauge image
        for gauge in gauge_list:
            clock_gauge_transform = InferenceUtils.albu_make_frame(
                img_shape=(640, 640), frame_coor=gauge.xyxy, target_size=(640, 640)
            )

            crop_frame_image = clock_gauge_transform(
                # image=torch.squeeze(gray_img, 0).numpy().transpose(1, 2, 0)
                image=original_image
            )[
                "image"
            ]  # tensor
            # ic(crop_frame_image.shape)
            
            #TODO: make crop image gray
            ic(crop_frame_image.shape)
            crop_frame_image_np = crop_frame_image.numpy().transpose(1,2,0)
            crop_frame_image_np = cv2.cvtColor(crop_frame_image_np, cv2.COLOR_RGB2GRAY)
            crop_frame_image_np = cv2.cvtColor(crop_frame_image_np , cv2.COLOR_GRAY2BGR)  # convert from 1 channel to 3 channel gray scale

            crop_frame_image = torch.from_numpy(crop_frame_image_np)
            crop_frame_image = torch.div(crop_frame_image, 255.0)
            crop_frame_image = np.transpose(crop_frame_image, (2, 0, 1))


            Utils.visualize_img_bb(
                img=crop_frame_image_np,
                bb=[],
                with_class=False,
                labels=None,
                format=None,
            )

            # Utils.visualize_img_bb(
            #     img=torch.squeeze(crop_frame_image, 0).numpy().transpose(1, 2, 0),
            #     bb=[],
            #     with_class=False,
            #     labels=None,
            #     format=None,
            # )

            # TODO: predict to get the value
            # real_clock_np = (
            #     torch.squeeze(crop_frame_image, 0).numpy().transpose(1, 2, 0)
            # )
            # real_clock_tensor = torch.div(crop_frame_image, 255.0)

            clock_result = model_clock_inside.inference_data(
                input_img=torch.unsqueeze(crop_frame_image, 0)
            )[0]

            detections = sv.Detections.from_ultralytics(clock_result).with_nms(
                threshold=0.5, class_agnostic=False
            )
            
            # ic(detections)

            detection_dict = {
                "gauge": None,
                "min": None,
                "max": None,
                "head": None,
                "center": None,
                "bottom": None,
            }

            detection_target_dict = {
                0:"gauge",
                1:"min",
                2:"max",
                3:"center",
                4:"head",
                5:"bottom",
            } 

            # ic(f"detection dict before: {detection_dict}")

            for detec_idx in range(len(detections.xyxy)):
                if (
                    # detection_dict[str(detections.data["class_name"][detec_idx])]
                    # is None
                    detection_dict[detection_target_dict[detections.class_id[detec_idx]]]
                    is None
                ):
                    detection_dict[detection_target_dict[detections.class_id[detec_idx]]] = {
                        "xyxy": detections.xyxy[detec_idx],
                        "class": detections.class_id[detec_idx],
                    }
            
            # ic(f"detection dict after: {detection_dict}")

            detection_dict_filter = {
                key: value for key, value in detection_dict.items() if value is not None
            }  # filter the detection

            # ic(detection_dict)

            clock_case = {
                "normal": "normal",
                "part": "part",
            }
            min_value = float(str(input("Please input min value: ")))
            max_value = float(str(input("Please input max value: ")))
            ic(min_value)
            ic(max_value)
            # min_value = 0.0
            # max_value = 100.0

            inference_clock = InferenceClock(
                case="normal",
                gauge=(
                    detection_dict["gauge"]["xyxy"]
                    if detection_dict["gauge"] is not None
                    else None
                ),
                min=(
                    detection_dict["min"]["xyxy"]
                    if detection_dict["min"] is not None
                    else None
                ),
                max=(
                    detection_dict["max"]["xyxy"]
                    if detection_dict["max"] is not None
                    else None
                ),
                center=(
                    detection_dict["center"]["xyxy"]
                    if detection_dict["center"] is not None
                    else None
                ),
                head=(
                    detection_dict["head"]["xyxy"]
                    if detection_dict["head"] is not None
                    else None
                ),
                bottom=(
                    detection_dict["bottom"]["xyxy"]
                    if detection_dict["bottom"] is not None
                    else None
                ),
                min_value=min_value,
                max_value=max_value,
            )

            # inference_clock.print()
            inference_clock.visualize_clock(image=torch.squeeze(crop_frame_image, 0).numpy().transpose(1, 2, 0))
            clock_value = inference_clock.predict_clock()
            ic(f"clock value: {clock_value}")


        if clock_index == 20:
            break

        # break

    end_time = time.time()
    ic(f"use time : {(end_time - start_time ) / number_samples}")
