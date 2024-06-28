# TODO: this file use to predict the model
from utils import Utils, Constants
from inference import (
    InferenceConstants,
    InferenceModel,
    Boxes,
    InferenceUtils,
    DigitalBoxes,
    NumberBoxes,
    ClockBoxes,
)
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image
from icecream import ic
import cv2
from PIL import Image
import time
import argparse
from pathlib import Path

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="for training arguments parser")

# Add positional argument
parser.add_argument("--input_file", type=str, help="your python file to run")
parser.add_argument("--gauge_use", type=str, help="type gauge")
parser.add_argument("--img_path", type=str, help="image path")

# # Add optional argument with a default value
parser.add_argument("--select_frame", type=int, default=0, help="selcet_frame from path and move to select_frame -> work only in number gauge type")

# Parse the command-line arguments
args = parser.parse_args()

gauge_use = Utils.get_enum_by_value(
    value=args.gauge_use.lower(), enum=Constants.GaugeType
)
img_path = args.img_path

image_size_dict = {
    Constants.GaugeType.digital: (640, 640),
    Constants.GaugeType.number: (640, 640),
    Constants.GaugeType.clock: (640,640),
}
model_path_dict = {
    Constants.GaugeType.digital: {
        "digital": Path("./models/digital/digital_model.pt"),
        "number": Path("./models/number/best (2).pt"),
    },
    Constants.GaugeType.number: Path("/home/kan.t/work/ABC_full_project/experiment/(ds_t)_number_(m_t)_m_3/weights/best.pt"),
    Constants.GaugeType.clock: Path("/home/kan/Desktop/Work/ABC_full_project/experiment/(ds_t)_clock_(m_t)_m/weights/best.pt")
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
        ) # not convert to gray scale image


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
                    image=torch.squeeze(crop_frame_image,0).numpy().transpose(1,2,0),
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
        Utils.delete_folder_mkdir(folder_path=Path("./select_frame"),remove=True)

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
                Utils.copy_file(source_file_path=frame_image_path, target_file_path=Path("./select_frame/"))


    end_time = time.time()
    ic(f"use time : {(end_time - start_time ) / number_samples}")

elif gauge_use == Constants.GaugeType.clock:
    print(f"--- Inference clock ---")
    
    model_clock = InferenceModel(
        model_path=model_path_dict[gauge_use], 
        img_target_size=image_size_dict[gauge_use],
        conf=0.25
    )
    
    samples = Utils.get_filenames_folder(
        source_folder=Path(img_path)
    )

    number_samples = len(samples)

    start_time =time.time()
    
    for clock_index, clock_img_path in enumerate(samples):
        print(f"clock index: {clock_index}")

        image = cv2.imread(str(clock_img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        trans = InferenceUtils.albu_resize_pad_zero(
            target_size=image_size_dict[gauge_use],
            gray=False,
        )

        image_tensor = trans(image=image, bboxes=[])["image"]
        image_tensor = torch.div(image_tensor, 255.0)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        print(f"image tensor shape: {image_tensor.shape}")

        clock_result = model_clock.inference_data(input_img=image_tensor)

        # Utils.visualize_img_bb(
        #     img=torch.squeeze(image_tensor, 0).numpy().transpose(1, 2, 0),
        #     bb=[],
        #     with_class=False,
        #     labels=None,
        #     format=None,
        # )
        # break

        for r in clock_result:
            nbp = r.boxes.cpu().numpy()
            # print(nbp)

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
                # image=torch.squeeze(image_tensor, 0).numpy().transpose(1, 2, 0),
                # image_orig=torch.squeeze(image_tensor, 0).numpy().transpose(1, 2, 0),
                # labels= ['gauge', "min", "max", "center", "head", "bottom"],
            )

            Utils.visualize_img_bb(
                # img=torch.squeeze(image_tensor, 0).numpy().transpose(1, 2, 0),
                img=torch.squeeze(image_tensor, 0).numpy().transpose(1, 2, 0),
                bb=[],
                with_class=True,
                labels=['gauge', "min", "max", "center", "head", "bottom"],
                format=Constants.BoundingBoxFormat.XYXY,
            )
            min_value = float(input("Please input min value: "))
            max_value = float(input("Please input max value: "))
            
            ic(boxes.predict_clock(gauge_min_value=min_value, gauge_max_value=max_value))

            Utils.visualize_img_bb(
                # img=torch.squeeze(image_tensor, 0).numpy().transpose(1, 2, 0),
                img=torch.squeeze(image_tensor, 0).numpy().transpose(1, 2, 0),
                bb=[
                    {"class": bb[5], "bb": [(bb[0], bb[1]), (bb[2], bb[3])]}
                    for bb in boxes.boxes
                ],
                with_class=True,
                labels=['gauge', "min", "max", "center", "head", "bottom"],
                format=Constants.BoundingBoxFormat.XYXY,
            )
        
        # if clock_index == 4:
        #     break
            
        # break

        # gauge_display_frames = number_model.inference_data(input_img=image_tensor)



    end_time = time.time()
    ic(f"use time : {(end_time - start_time ) / number_samples}")
