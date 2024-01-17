from utils import Utils, Constants, ConfusionMatrix, IOU, mAP, AP, DetMetrics
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
import argparse

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="for validating the model and dataset")

# Add positional argument
parser.add_argument("input_file",type=str, help = "your python file to run, only add not use")
parser.add_argument("gauge_type", type=str, help="for loading labels name and index")
parser.add_argument("model_path", type=str, help="path to your model")
parser.add_argument("dataset_path", type=str, help="path to your dataset that has images and labels folder")
# parser.add_argument("", type=str, help="model type to validate")


# # Add optional argument with a default value
parser.add_argument("-p", "--plot", type=bool, default=False, help="plot a confusion matrix and r,p,f1,map curve or not")
# parser.add_argument("-imgs", "--img_size", type=int, choices=[640,1024], default=1024, help="size of train images")
# parser.add_argument("-bs", "--batch_size", type=int, default=16, help="number of batch size")
# parser.add_argument("-c", "--cache",type=bool, default=False, help="cache the image for True and False")
# parser.add_argument("-p", "--patience", type=int, default=20, help="set number of how many not learning epochs to stop training")
# parser.add_argument("-d", "--device",type=str,choices=["cpu","mps","0","1"], default="cpu", help="choose device to train")
# parser.add_argument("-w", "--workers", type=int, default=12 , help="set the number of workers")
# parser.add_argument("-rs", "--resume", type=bool, default=False, help="resume training")
# parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate")
# parser.add_argument("-flr", "--final_learning_rate", type=float,default=0.01, help="final learning rate")

# Parse the command-line arguments
args = parser.parse_args()

# gauge_use = Constants.GaugeType.digital
gauge_use = Utils.get_enum_by_value(value=args.gauge_type.lower(),enum=Constants.GaugeType)
# is_plot = False
is_plot = args.plot

# TODO: load the model
# model_path = Path("./models/digital/digital_model.pt")
model_path = Path(args.model_path)
model = InferenceModel(model_path=model_path, img_target_size=(640, 640), conf=0.25)

# TODO: prepare test data
# dataset_path = Path(f"./datasets_for_train/{gauge_use.value}/test/")
dataset_path = Path(args.dataset_path)
dataset_img_bb = Utils.match_img_bb_filename(
    img_filenames_list=Utils.get_filenames_folder(dataset_path / "images"),
    bb_filenames_list=None,
    source_folder=dataset_path,
)

true_bb = []
pred_bb = []

label_class = Constants.map_data_dict[gauge_use.value]["target"]
iou_list = torch.linspace(0.5, 0.95, 10)
confidence = 0.25
confusion_matrix_dict = {
    _iou: ConfusionMatrix(
        num_classes=len(label_class),
        iou_threshold=_iou,
        conf=confidence,
        labels=label_class,
    )
    for _iou in iou_list
}

detection_matrics = DetMetrics(
    names=label_class,
    conf=confidence,
    # iou_threshold=0.5,
    save_dir=dataset_path,
    plot=is_plot,
)

# TODO: inference model
for idx, (img_path, bb_path) in enumerate(dataset_img_bb):
    image = cv2.imread(str(img_path))
    trans = InferenceUtils.albu_resize_pad_zero(
        target_size=(640, 640),
    )

    image_tensor = trans(image=image, bboxes=[])["image"]
    image_tensor = torch.div(image_tensor, 255.0)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output = model.inference_data(input_img=image_tensor)

    # _true_bb format = x1,y1,x2,y2,class
    _true_bb = Utils.load_bb(filepath=str(bb_path))
    _gt_cls = torch.tensor([])

    if _true_bb is not None:
        _gt_cls = torch.tensor(
            [int(_t_bb[0]) for _t_bb in _true_bb]
        )  # gt class for update confusion matrix
        _true_bb = [
            Utils.change_format_yolo2xyxy(
                img_size=(640, 640), bb=t_bb, with_class=True
            )["bb"]
            for t_bb in _true_bb
        ]  # return [(x1,y1), (x2,y2)] format
        _true_bb = torch.tensor(
            [[_t_bb[0][0], _t_bb[0][1], _t_bb[1][0], _t_bb[1][1]] for _t_bb in _true_bb]
        )  # return [x1,y1,x2,y2] format

    # _pred_bb format [x1,y1,x2,y2,conf,class]
    _pred_bb = None
    if output[0].boxes.cls.size(0) > 0:
        _pred_bb = torch.cat(
            (
                output[0].boxes.xyxy,
                torch.unsqueeze(output[0].boxes.conf, dim=1),
                torch.unsqueeze(output[0].boxes.cls, dim=1),
            ),
            dim=1,
        )

    for conf_v in confusion_matrix_dict.keys():
        confusion_matrix_dict[conf_v].update(
            gt_boxes=_true_bb, detections=_pred_bb, gt_classes=_gt_cls
        )

    detection_matrics.collect_results(
        gt_boxes=_true_bb, gt_classes=_gt_cls, pred_boxes=_pred_bb
    )

detection_matrics.post_collected_data()
detection_matrics.process()  # including plot image for P_curve, R_curve , PR_curve
# ic(detection_matrics.keys)
# ic(detection_matrics.mean_results())
# ic(detection_matrics.class_result(0))
ic(detection_matrics.maps)
ic(detection_matrics.box.ap)
ic(detection_matrics.box.map50)
ic(detection_matrics.box.map75)
# ic(detection_matrics.fitness)
# ic(detection_matrics.ap_class_index)
# ic(detection_matrics.results_dict)
# ic(detection_matrics.curves)
# ic(detection_matrics.curves_results)

for conf_v in confusion_matrix_dict.keys():
    ic(f"--- Confusion matrix at iou { (conf_v.item()):.2f}")
    confusion_matrix_dict[conf_v].print()
    print()
    if is_plot:
        confusion_matrix_dict[conf_v].plot(normalize=False, save_dir=dataset_path)
