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

gauge_use = Constants.GaugeType.digital

# TODO: load the model
model_path = Path("./models/digital/digital_model.pt")
model = InferenceModel(model_path=model_path, img_target_size=(640, 640), conf=0.25)

# TODO: prepare test data
dataset_path = Path(f"./datasets_for_train/{gauge_use.value}/test/")
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
    _iou : ConfusionMatrix(
        num_classes=len(label_class), iou_threshold=_iou, conf=confidence, labels=label_class
    )
    for _iou in iou_list
}

detection_matrics = DetMetrics(
    names=label_class,
    conf=confidence,
    # iou_threshold=0.5,
    save_dir=dataset_path,
    plot=True,
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
        confusion_matrix_dict[conf_v].update(gt_boxes=_true_bb, detections=_pred_bb, gt_classes=_gt_cls)

    detection_matrics.collect_results(gt_boxes=_true_bb,gt_classes=_gt_cls, pred_boxes=_pred_bb)

ic(detection_matrics.collected_data)
# for i, co in enumerate(detection_matrics.collected_data):
#     ic(f"iou {0.5 + 0.05*i} : number of bb {co.size(0)}")
# detection_matrics.process() #including plot image for P_curve, R_curve , PR_curve
# ic(detection_matrics.keys)
# ic(detection_matrics.mean_results())
# ic(detection_matrics.class_result(0))
# ic(detection_matrics.maps)
# ic(detection_matrics.fitness)
# ic(detection_matrics.ap_class_index)
# ic(detection_matrics.results_dict)
# ic(detection_matrics.curves)
# ic(detection_matrics.curves_results)

for conf_v in confusion_matrix_dict.keys():
    confusion_matrix_dict[conf_v].print()
    print()
    confusion_matrix_dict[conf_v].plot(normalize=False, save_dir=dataset_path)
    