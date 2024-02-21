from icecream import ic
from inference.inference_constants import InferenceConstants
from inference.inference_utils import InferenceUtils
from enum import Enum, auto
from utils import Utils, KMeanClutering
import math
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
from sklearn.cluster import KMeans
import numpy as np


class Boxes:
    def __init__(
        self,
        boxes,
        cls,
        conf,
        data,
        id,
        is_track,
        orig_shape,
        shape,
        xywh,
        xywhn,
        xyxy,
        xyxyn,
    ):
        self.boxes = boxes
        self.cls = cls
        self.conf = conf
        self.data = data
        self.id = id
        self.is_track = is_track
        self.orig_shape = orig_shape
        self.shape = shape
        self.xywh = xywh
        self.xywhn = xywhn
        self.xyxy = xyxy
        self.xyxyn = xyxyn

        self.boxes_list = [
            Box(
                box=self.boxes[index],
                cls=self.cls[index],
                conf=self.conf[index],
                data=self.data[index],
                xywh=self.xywh[index],
                xywhn=self.xywhn[index],
                xyxy=self.xyxy[index],
                xyxyn=self.xyxyn[index],
            )
            for index, value in enumerate(self.boxes)
        ]

    def makeBBForSave(
        self,
    ):
        bb_save = []
        for index, box in enumerate(self.boxes):
            box_class = int(self.boxes[index][-1])
            yolo_format_bb = Utils.change_xyxy_to_yolo(
                xyxy_format=self.boxes[index][:-2],
                class_bb=box_class,
                image_width=self.orig_shape[1],
                image_height=self.orig_shape[0],
            )
            # ic(yolo_format_bb)
            bb_save.append(yolo_format_bb)
        return bb_save

    class COORDINATES(Enum):
        X1 = auto()
        Y1 = auto()
        X2 = auto()
        Y2 = auto()

    # def getCoordinatesRealImage(
    #     self, ori_shape, want_shape, box_coor
    # ):  # convert cooridates from convert_shape to ori_shape
    #     # ic(ori_shape, want_shape, box_coor)
    #     ori_w = ori_shape[1]
    #     ori_h = ori_shape[0]

    #     want_w = want_shape[1]
    #     want_h = want_shape[0]

    #     box_c = {
    #         self.COORDINATES.X1: math.floor(box_coor[0]),
    #         self.COORDINATES.Y1: math.floor(box_coor[1]),
    #         self.COORDINATES.X2: math.ceil(box_coor[2]),
    #         self.COORDINATES.Y2: math.ceil(box_coor[3]),
    #     }

    #     new_x1 = self.NewRangRatio(
    #         ori_range=ori_w, new_range=want_w, ori_value=box_c[self.COORDINATES.X1]
    #     )
    #     new_x2 = self.NewRangRatio(
    #         ori_range=ori_w, new_range=want_w, ori_value=box_c[self.COORDINATES.X2]
    #     )
    #     new_y1 = self.NewRangRatio(
    #         ori_range=ori_h, new_range=want_h, ori_value=box_c[self.COORDINATES.Y1]
    #     )
    #     new_y2 = self.NewRangRatio(
    #         ori_range=ori_h, new_range=want_h, ori_value=box_c[self.COORDINATES.Y2]
    #     )

    #     # ic(f"ori_x1: {box_c[self.COORDINATES.X1]}, new_x1: {new_x1}")
    #     # ic(f"ori_x2: {box_c[self.COORDINATES.X2]}, new_x1: {new_x2}")
    #     # ic(f"ori_y2: {box_c[self.COORDINATES.Y1]}, new_x1: {new_y1}")
    #     # ic(f"ori_x1: {box_c[self.COORDINATES.Y2]}, new_x1: {new_y2}")

    #     return [new_x1, new_y1, new_x2, new_y2]

    # def NewRangRatio(self, ori_range, new_range, ori_value):
    #     return int((ori_value * new_range) / ori_range)

    def getCenter(self, minimum, maximum):
        return int(((maximum - minimum) / 2) + minimum)


class Box:
    def __init__(
        self,
        box,
        cls,
        conf,
        data,
        xywh,
        xywhn,
        xyxy,
        xyxyn,
        mean=None, # ues for predict number,
        group=None, # use for k mean
    ):
        self.box = box
        self.cls = cls
        self.conf = conf
        self.data = data
        self.xywh = xywh
        self.xywhn = xywhn
        self.xyxy = xyxy
        self.xyxyn = xyxyn
        self.mean = mean # use for predict number 
        self.group = group # use for k mean

    def printBox(
        self,
    ):
        ic(f"box: {self.box}")
        ic(f"cls: {self.cls}")
        ic(f"conf: {self.conf}")
        ic(f"data: {self.data}")
        ic(f"xywh: {self.xywh}")
        ic(f"xywhn: {self.xywhn}")
        ic(f"xyxy: {self.xyxy}")
        ic(f"xyxyn: {self.xyxyn}")
        ic(f"mean: {self.mean}") # use for predict number
        ic(f"group: {self.group}")
        print()


class DigitalBoxes(Boxes):
    def __init__(
        self,
        boxes,
        cls,
        conf,
        data,
        id,
        is_track,
        orig_shape,
        shape,
        xywh,
        xywhn,
        xyxy,
        xyxyn,
    ):
        super().__init__(
            boxes=boxes,
            cls=cls,
            conf=conf,
            data=data,
            id=id,
            is_track=is_track,
            orig_shape=orig_shape,
            shape=shape,
            xywh=xywh,
            xywhn=xywhn,
            xyxy=xyxy,
            xyxyn=xyxyn,
        )

        self.gaugeList = self.makeGaugesList()
        self.displayList = self.makeDisplayList()
        self.frameList = self.makeFrameList()

        self.nGauges = len(self.gaugeList)
        self.nDisplays = len(self.displayList)
        self.nFrames = len(self.frameList)

        self.frameInGauge = self.frameInGauge()
        self.frameInDisplay = self.frameInDisplay()

        self.framePredict = self.makeFrameForPredict()

    def makeFrameForPredict(
        self,
    ):
        predict_frame_index = []  # contain index of frame use for prediction
        """
        step 1: check frame in display
        """
        if self.nDisplays != 0:
            for display in self.displayList:
                for frame_index, frame in enumerate(self.frameList):
                    if InferenceUtils.is_overlapping(
                        bbox1=display.xyxy, bbox2=frame.xyxy
                    ):
                        predict_frame_index.append(frame_index)

        """
        step 2: check gauge overlap display
        """
        if self.nGauges != 0:
            for gauge in self.gaugeList:
                is_overlap = False
                for display in self.displayList:
                    if InferenceUtils.is_overlapping(
                        bbox1=gauge.xyxy, bbox2=display.xyxy
                    ):
                        is_overlap = True
                        break

                if not is_overlap:
                    for frame_index, frame in enumerate(self.frameList):
                        if InferenceUtils.is_overlapping(
                            bbox1=gauge.xyxy, bbox2=frame.xyxy
                        ):
                            predict_frame_index.append(frame_index)

        """
        step3 : if the image have only frame        
        """
        if len(predict_frame_index) == 0:
            return self.frameList

        """
        step 4: convert predict_frame_index to frame  
        """
        predict_frame = []
        for frame_index in set(predict_frame_index):
            predict_frame.append(self.frameList[frame_index])

        return predict_frame

    def makeGaugesList(
        self,
    ):
        gauge_list = [
            b
            for b in self.boxes_list
            if int(b.cls) == InferenceConstants.inference_class_dict["gauge"]
        ]
        return gauge_list

    def makeDisplayList(
        self,
    ):
        display_list = [
            b
            for b in self.boxes_list
            if int(b.cls) == InferenceConstants.inference_class_dict["display"]
        ]
        return display_list

    def makeFrameList(
        self,
    ):
        frame_list = [
            b
            for b in self.boxes_list
            if int(b.cls) == InferenceConstants.inference_class_dict["frame"]
        ]
        return frame_list

    def frameInGauge(
        self,
    ):
        new_frame = []
        for gauge in self.gaugeList:
            for frame in self.frameList:
                if InferenceUtils.is_overlapping(bbox1=gauge.xyxy, bbox2=frame.xyxy):
                    new_frame.append(frame)

        return new_frame

    def frameInDisplay(
        self,
    ):
        new_frame = []
        for display in self.displayList:
            for frame in self.frameList:
                if InferenceUtils.is_overlapping(bbox1=display.xyxy, bbox2=frame.xyxy):
                    new_frame.append(frame)

        return new_frame


class NumberBoxes(Boxes):
    def __init__(
        self,
        boxes,
        cls,
        conf,
        data,
        id,
        is_track,
        orig_shape,
        shape,
        xywh,
        xywhn,
        xyxy,
        xyxyn,
        image=None,
    ):
        super().__init__(
            boxes=boxes,
            cls=cls,
            conf=conf,
            data=data,
            id=id,
            is_track=is_track,
            orig_shape=orig_shape,
            shape=shape,
            xywh=xywh,
            xywhn=xywhn,
            xyxy=xyxy,
            xyxyn=xyxyn,
        )
        self.image = image

    def predict_number(
        self,
    ):
        float_cls = 12
        dot_cls = 11
        

        # convert cls to number
        sort_boxes = self.sort_boxes()

        is_float = False
        is_dot = False
        # ic(sort_boxes)
        # ic(self.boxes_list)

        ans = ""
        for _, cls in sort_boxes.items():
            if int(cls) == float_cls:
                is_float = True
                # ic(f"found float")
            
            if int(cls) == dot_cls:
                is_dot = True
                # ic(f"found dot")

            ans += InferenceConstants.inference_number_convert[cls]
        
        if (is_float and (not is_dot)) or (is_float and is_dot):
            ans = ""
            float_boxes = []
            for box in self.boxes_list:
                if int(box.cls) == float_cls:
                    float_boxes.append(box) 
            
            # ic(f"number of float boxes : {len(float_boxes)}")
            float_start = float_boxes[0].xyxy[0]

            for center_x_pos,cls in sort_boxes.items():
                if (int(cls) == float_cls) or (int(cls) == dot_cls):
                    continue
                
                # ic(ans, float_start, center_x_pos)
                if (center_x_pos > float_start) and (ans.find(".") == -1): # ans doesn't have . 
                    ans += "."
                
                 
                ans += InferenceConstants.inference_number_convert[cls]
            
        
        if len(self.boxes_list) and ans[-1] == ".": # for ex 12. -> 12.0
            ans += "0"

        # # TODO: step check dot class in boxes
        # class_list = [int(box.cls) for box in self.boxes_list]
        # dot_class = 11  # see in inferenceConstants
        # ans = ""
        
        # if not len(self.boxes_list) :  # can not detect number bounding box
        #     return "0.0"

        # if (dot_class in class_list) or len(self.boxes_list) <= 2:  # found dot class
        #     # convert cls to number
        #     sort_boxes = self.sort_boxes()
        #     ans = ""
        #     for _, cls in sort_boxes.items():
        #         ans += InferenceConstants.inference_number_convert[cls]
            
        #     if ans[-1] == ".": # for ex 12. -> 12.0
        #         ans += "0"
        # else:

        #     integer_number = [] #store interger number xxxx.00
        #     float_number = [] #store float number 00.xxx
        #     box_mean_list = []

        #     for box in self.boxes_list:
        #         transform = A.Compose(
        #             [
        #                 A.Crop(
        #                     x_min=max(int(box.xyxy[0]),0),
        #                     y_min=max(int(box.xyxy[1]),0),
        #                     x_max=min(int(box.xyxy[2]), self.image.shape[1]-1),
        #                     y_max=min(int(box.xyxy[3]), self.image.shape[0]-1),
        #                 ),
        #                 ToTensorV2(),
        #             ]
        #         )
        #         image_crop = transform(image=self.image)["image"]
        #         mean_img = torch.mean(image_crop * 255, dim=(0,1,2))
        #         box.mean = int(mean_img.item())
        #         box_mean_list.append(box.mean)

        #     k_mean = KMeans(n_clusters=2, random_state=0).fit(np.array(box_mean_list).reshape(-1,1))
        #     for k_idx, group  in enumerate(k_mean.labels_):
        #         self.boxes_list[k_idx].group = group
            
        #     cluster_center = k_mean.cluster_centers_
        #     mean = np.mean(np.array(box_mean_list))
        #     ic(cluster_center)
        #     ic(mean)

        #     if (abs(cluster_center[0] - mean) > 20) or (abs(cluster_center[1] - mean) > 20):
        #         sort_boxes = self.sort_boxes()
        #         for _, sb_value in sort_boxes.items():
        #             box_group = sb_value[1]
        #             box_value = InferenceConstants.inference_number_convert[sb_value[0]]
        #             if int(box_group) == 0: # integer number
        #                 integer_number.append(box_value)
        #             else: #float number
        #                 float_number.append(box_value) 

        #         ans = "" 
        #         if len(integer_number) > 0:
        #             for n in integer_number:
        #                 ans += n
        #         else:
        #             ans += "0"
                
        #         if len(float_number) > 0:
        #             ans += "."
        #             for f in float_number:
        #                 ans += f
        #         else:
        #             ans += ".0"
        #     else:
        #         sort_boxes = self.sort_boxes()
        #         ans = ""
        #         for center, (cls,_) in sort_boxes.items():
        #             ans += InferenceConstants.inference_number_convert[cls]
                
        #         if ans[-1] == ".": # for ex 123. -> 123.0
        #             ans += "0"

        if ans == "":
            ans = "0.0"
        
        ic(ans)
        # ic(self.image.shape)
        # ic(self.image.dtype)
        # Utils.visualize_img_bb(
        #         img=self.image,
        #         bb=[],
        #         with_class=False,
        #         labels=None,
        #         format=None,
        #     )

        return ans

    def sort_boxes(
        self,
    ):
        # get dictionary containing center of x
        center_dict = {
            self.getCenter(minimum=box.xyxy[0], maximum=box.xyxy[2]): box.cls if box.group is None else (box.cls, box.group)
            for index, box in enumerate(self.boxes_list)
        }

        #if found dot class return only class number
        #if not found dot class return class number, group number

        center_dict = dict(sorted(center_dict.items()))
        return center_dict  # return dict that key=center, value=cls
