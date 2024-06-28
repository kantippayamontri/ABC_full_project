import math
from enum import Enum, auto
from math import atan

import albumentations as A
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from icecream import ic
from sklearn.cluster import KMeans
from sympy import solve
from sympy.abc import x, y

from inference.inference_constants import InferenceConstants
from inference.inference_utils import InferenceUtils
from utils import Constants, KMeanClutering, Utils


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
        mean=None,  # ues for predict number,
        group=None,  # use for k mean
    ):
        self.box = box
        self.cls = cls
        self.conf = conf
        self.data = data
        self.xywh = xywh
        self.xywhn = xywhn
        self.xyxy = xyxy
        self.xyxyn = xyxyn
        self.mean = mean  # use for predict number
        self.group = group  # use for k mean

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
        ic(f"mean: {self.mean}")  # use for predict number
        ic(f"group: {self.group}")
        print()

    def get_center_xyxy(
        self,
    ):
        x1, y1, x2, y2 = self.xyxy

        return (x1 + (x2 - x1) / 2.0, y1 + (y2 - y1) / 2.0)


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

            for center_x_pos, cls in sort_boxes.items():
                if (int(cls) == float_cls) or (int(cls) == dot_cls):
                    continue

                # ic(ans, float_start, center_x_pos)
                if (center_x_pos > float_start) and (
                    ans.find(".") == -1
                ):  # ans doesn't have .
                    ans += "."

                ans += InferenceConstants.inference_number_convert[cls]

        if len(self.boxes_list) and ans[-1] == ".":  # for ex 12. -> 12.0
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
            self.getCenter(minimum=box.xyxy[0], maximum=box.xyxy[2]): (
                box.cls if box.group is None else (box.cls, box.group)
            )
            for index, box in enumerate(self.boxes_list)
        }

        # if found dot class return only class number
        # if not found dot class return class number, group number

        center_dict = dict(sorted(center_dict.items()))
        return center_dict  # return dict that key=center, value=cls


class ClockBoxes(Boxes):
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
        # for debug
        # image_orig,
        # labels
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

        # self.image_orig = image_orig
        # self.labels = labels

        (
            self.gauge_list,
            self.min_list,
            self.max_list,
            self.center_list,
            self.head_list,
            self.bottom_list,
        ) = ic(self.make_bb_list(boxes_list=self.boxes_list))

    def make_bb_list(self, boxes_list):
        _gauge_list = []
        _min_list = []
        _max_list = []
        _center_list = []
        _head_list = []
        _bottom_list = []

        for b in boxes_list:
            if int(b.cls) == InferenceConstants.inference_clock_dict["gauge"]:
                _gauge_list.append(b)
            elif int(b.cls) == InferenceConstants.inference_clock_dict["min"]:
                _min_list.append(b)
            elif int(b.cls) == InferenceConstants.inference_clock_dict["max"]:
                _max_list.append(b)
            elif int(b.cls) == InferenceConstants.inference_clock_dict["center"]:
                _center_list.append(b)
            elif int(b.cls) == InferenceConstants.inference_clock_dict["head"]:
                _head_list.append(b)
            elif int(b.cls) == InferenceConstants.inference_clock_dict["bottom"]:
                _bottom_list.append(b)

        return _gauge_list, _min_list, _max_list, _center_list, _head_list, _bottom_list

    def predict_clock(
        self,
        gauge_min_value,
        gauge_max_value,
    ):
        # check gauge
        if len(self.gauge_list) > 0:

            gauge_conf_list = list([_gauge.conf for _gauge in self.gauge_list])
            gauge_index_max_conf = np.argmax(gauge_conf_list)
            gauge_use = self.gauge_list[gauge_index_max_conf]

            min_use = None
            max_use = None
            center_use = None
            head_use = None
            bottom_use = None

            # check min is in gauge:
            min_use = self.box_in_gauge(
                gauge_xyxy=gauge_use.xyxy, check_list=self.min_list
            )

            # check max is in gauge:
            max_use = self.box_in_gauge(
                gauge_xyxy=gauge_use.xyxy, check_list=self.max_list
            )

            # check center is in gauge
            center_use = self.box_in_gauge(
                gauge_xyxy=gauge_use.xyxy, check_list=self.center_list
            )

            # check head in gauge:
            head_use = self.box_in_gauge(
                gauge_xyxy=gauge_use.xyxy, check_list=self.head_list
            )

            # check bottom is in gauge
            bottom_use = self.box_in_gauge(
                gauge_xyxy=gauge_use.xyxy, check_list=self.bottom_list
            )

            # --------------------------------------Preprocess 6 class---------------------------------------------

            # check is found min and max
            if (min_use is not None) and (max_use is not None):
                ...
            else:
                return 99

            # check is found head center and bottom
            if (
                (center_use is not None)
                and (head_use is not None)
                and (bottom_use is not None)
            ):
                ...
            else:
                return 99

            # --------------------------------------Predict gauge value-----------------------------------------------

            clock_ratio = self.predict_clock_value(
                gauge=gauge_use,
                min=min_use,
                max=max_use,
                center=center_use,
                head=head_use,
                bottom=bottom_use,
            )
            
            range_value = gauge_max_value - gauge_min_value
            actual_value = clock_ratio * range_value + gauge_min_value

            return actual_value

        else:

            return 99

        return 0.0

    def box_in_gauge(self, gauge_xyxy, check_list):
        temp_use = []
        if len(check_list) == 0:
            return None
        else:
            for _check in check_list:
                if InferenceUtils.is_overlapping(bbox1=gauge_xyxy, bbox2=_check.xyxy):
                    temp_use.append(_check)

        if len(temp_use) == 0:
            return None
        elif len(temp_use) == 1:
            return temp_use[0]
        else:
            _conf_list = list([box.conf for box in temp_use])
            max_conf_index = np.argmax(_conf_list)
            return temp_use[max_conf_index]

        return None

    def predict_clock_value(
        self, gauge: Box, min: Box, max: Box, center: Box, head: Box, bottom: Box
    ):
        # self.visualize_clock_img_bb(
        #     img=self.image_orig,
        #     bb =[{"class": int(gauge.cls), "bb":[(gauge.xywh[0],gauge.xywh[1],gauge.xywh[2],gauge.xywh[3],)]}],
        #     labels=self.labels,
        #     with_class=True,
        #     format=Constants.BoundingBoxFormat.YOLOV8,

        # )
        # center_min_slope= self.get_slope_from_2_point(point1=center.get_center_xyxy(), point2=min.get_center_xyxy())
        # print(f"center_min_line: {center_min_slope}")
        # center_head_slope= self.get_slope_from_2_point(point1=center.get_center_xyxy(), point2=head.get_center_xyxy())
        # print(f"center_head_line: {center_head_slope}")
        # center_max_slope= self.get_slope_from_2_point(point1=center.get_center_xyxy(), point2=max.get_center_xyxy())
        # print(f"center max line: {center_max_slope}")

        # min_head_angle = self.findAngle(M1=center_min_slope, M2=center_head_slope)
        # min_max_angle = self.findAngle(M1=center_min_slope, M2=center_max_slope)

        # print(f"min_head_angle: {min_head_angle}")
        # print(f"min_max_angle: {min_max_angle}")

        # # TODO: create circle
        # circle_center = center.get_center_xyxy() # center point of the circle
        # radius = self.get_distance_from_2_point(point1=min.get_center_xyxy(), point2=max.get_center_xyxy()) / 2.0
        # circle_eq = (circle_center[0] - x)**2 + (circle_center[1] - y)**2 - radius**2

        # print(f"radius: {radius}")
        # print(f"min to center l: {self.get_distance_from_2_point(point1=center.get_center_xyxy(), point2=min.get_center_xyxy())}")
        # print(f"max to center l: {self.get_distance_from_2_point(point1=center.get_center_xyxy(), point2=max.get_center_xyxy())}")

        # # find min coor in the circle
        # min_m, min_b = self.get_slope_intercept(point1=center.get_center_xyxy(), point2=min.get_center_xyxy())

        # ic(solve([(circle_center[0] - x)**2 + (circle_center[1] - y)**2 - radius**2,min_m*x + min_b], [x,y], dict=True))
        # # min_x_circle, min_y_circle = self.find_xy_circle(circle_eq=circle_eq, line_m=min_m, line_b=min_b)

        image_height = self.orig_shape[0]
        image_width = self.orig_shape[1]
        # origin_point = (image_width/2.0, image_height/2.0)
        origin_point = center.get_center_xyxy()  # set center point to the origin_point

        center_origin = self.set_point_2_origin(
            origin=origin_point, point=center.get_center_xyxy()
        )
        min_origin = self.set_point_2_origin(
            origin=origin_point, point=min.get_center_xyxy()
        )
        max_origin = self.set_point_2_origin(
            origin=origin_point, point=max.get_center_xyxy()
        )
        head_origin = self.set_point_2_origin(
            origin=origin_point, point=head.get_center_xyxy()
        )
        bottom_origin = self.set_point_2_origin(
            origin=origin_point, point=bottom.get_center_xyxy()
        )

        ic(min_origin)
        ic(max_origin)
        ic(head_origin)
        ic(bottom_origin)
        ic(center_origin)

        # angle_min_head = self.angle_between_2_line(a=min_origin, b=head_origin)
        # ic(angle_min_head)

        # angle_min_max = self.angle_between_2_line(a=min_origin,b=max_origin)
        # # check min,max y position with y position of center
        # if min_origin[1] < 0 and max_origin[1] <0:
        #     angle_min_max = 360 - angle_min_max

        # ic(angle_min_max)

        # min_head = np.array(min_origin)  # start from min and rotate to head
        # min_head = self.rotation_vector(theta=np.deg2rad(30), vector=min_head)

        # for deg in range(10, 360, 10):
        #     _min_head = self.rotation_vector(theta=np.deg2rad(deg), vector=min_head)
        #     vectors = [_min_head]
        #     ic(vectors)

        #     self.visualize_clock(
        #         gauge=gauge,
        #         min=min,
        #         max=max,
        #         center=center,
        #         head=head,
        #         bottom=bottom,  # clock_center=center.get_center_xyxy(), clock_r=radius,
        #         vectors=list(
        #             [
        #                 [
        #                     origin_point[0],
        #                     origin_point[1],
        #                     v[0] + origin_point[0],
        #                     -1 * v[1] + origin_point[1],
        #                 ]
        #                 for v in vectors
        #             ]
        #         ),
        #     )

        # ic(vectors)

        
        head_min_angle, head_min_v = self.angle_between_2_vector(start_vector=head_origin,end_vector=min_origin)
        ic(head_min_angle)
        
        max_head_angle, max_head_v = self.angle_between_2_vector(start_vector=max_origin, end_vector=head_origin, )
        ic(max_head_angle)
        
        all_angle = head_min_angle + max_head_angle
        ratio = head_min_angle / all_angle
        
        
        
        
        vectors = [head_min_v, max_head_v]
        self.visualize_clock(
            gauge=gauge,
            min=min,
            max=max,
            center=center,
            head=head,
            bottom=bottom,  # clock_center=center.get_center_xyxy(), clock_r=radius,
            vectors=list([ [origin_point[0], origin_point[1], v[0] + origin_point[0], -1*v[1] + origin_point[1] ] for v in vectors]),
        )
        return ratio

    def angle_between_2_vector(
        self, start_vector, end_vector, step_deg=1, min_deg_thres=0
    ):
        temp_vector = start_vector
        for deg in range(0, 360, step_deg):
            if (
                int(self.angle_between_2_line(a=temp_vector, b=end_vector))
                > min_deg_thres
            ):
                temp_vector = self.rotation_vector(theta=np.deg2rad(deg), vector=start_vector)
            else:
                return deg , temp_vector
                

    def rotation_vector(self, theta, vector):  # theta in radians
        rotation_matrix = np.array(
            [
                [math.cos(theta), -1 * math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ]
        )

        vector_use = np.array(vector)

        # print(f"rotation matrix shape: {rotation_matrix.shape}")
        # print(f"vector_use shape: {vector_use.shape}")

        rotated_vector = rotation_matrix @ vector_use

        return rotated_vector

    def angle_between_2_line(self, a, b):
        # convert a and b from tuple to numpy array
        vec_a = np.array(a)
        vec_b = np.array(b)

        dot_ab = vec_a @ vec_b
        magnitude_a = self.get_distance_from_2_point(point1=(0, 0), point2=a)
        magnitude_b = self.get_distance_from_2_point(point1=(0, 0), point2=b)

        rad = np.arccos(dot_ab / (magnitude_a * magnitude_b))
        degree = np.rad2deg(rad)

        return degree

    def set_point_2_origin(self, origin, point):
        return (point[0] - origin[0], -1 * (point[1] - origin[1]))

    # def find_xy_circle(self, circle_eq, line_m,line_b):
    #     line_eq = line_m * x + line_b - y
    #     ic(solve([circle_eq, line_eq],[x,y],dict=True))

    def get_distance_from_2_point(self, point1: tuple, point2: tuple):
        """
        point1: tuple (x,y)
        point2: tuple (x,y)
        """
        distance = math.sqrt(
            (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        )
        return distance

    # def findAngle(self,M1, M2):
    #     PI = 3.14159265

    #     # Store the tan value  of the angle
    #     angle = abs((M2 - M1) / (1 + M1 * M2))

    #     # Calculate tan inverse of the angle
    #     ret = atan(angle)

    #     # Convert the angle from
    #     # radian to degree
    #     val = (ret * 180) / PI

    #     # Print the result
    #     return (round(val, 4))

    def get_slope_intercept(self, point1, point2):
        slope = self.get_slope(point1=point1, point2=point2)
        y_intercept = self.get_y_intercept(slope=slope, point=point1)
        return slope, y_intercept

    def get_slope(self, point1, point2):
        # calculate m from deltaY/deltaX
        delta_y = (640 - point2[1]) - (640 - point1[1])
        delta_x = point2[0] - point1[0]
        return delta_y / delta_x

    def get_y_intercept(self, slope, point):
        return point[1] - slope * point[0]

    def visualize_clock(
        self,
        gauge: Box,
        min: Box,
        max: Box,
        center: Box,
        head: Box,
        bottom: Box,
        image_size=(640, 640),
        vectors: list = [],
    ):
        # create mock images
        image = np.ones(image_size)
        print(f"center: {center.get_center_xyxy()}")
        print(f"gauge: {gauge.get_center_xyxy()}")
        print(f"min: {min.get_center_xyxy()}")
        print(f"max: {max.get_center_xyxy()}")

        plt.imshow(image)
        gauge_x1, gauge_y1, gauge_x2, gauge_y2 = gauge.xyxy
        plt.gca().add_patch(
            patches.Rectangle(
                xy=(gauge_x1, gauge_y1),
                width=gauge_x2 - gauge_x1,
                height=gauge_y2 - gauge_y1,
                linewidth=2,
                edgecolor=np.array([0, 255, 0]) / 255.0,
                facecolor="none",
            )
        )

        min_x, min_y = min.get_center_xyxy()
        plt.plot(min_x, min_y, "ro", markersize=10)
        plt.annotate(
            "min",
            (min_x, min_y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

        max_x, max_y = max.get_center_xyxy()
        plt.plot(max_x, max_y, "ro", markersize=10)
        plt.annotate(
            "max",
            (max_x, max_y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

        center_x, center_y = center.get_center_xyxy()
        plt.plot(center_x, center_y, "ro", markersize=10)
        plt.annotate(
            "center",
            (center_x, center_y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

        head_x, head_y = head.get_center_xyxy()
        plt.plot(head_x, head_y, "ro", markersize=10)
        plt.annotate(
            "head",
            (head_x, head_y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

        bottom_x, bottom_y = bottom.get_center_xyxy()
        plt.plot(bottom_x, bottom_y, "ro", markersize=10)
        plt.annotate(
            "bottom",
            (bottom_x, bottom_y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

        plt.plot(
            (center.get_center_xyxy()[0], min.get_center_xyxy()[0]),
            (center.get_center_xyxy()[1], min.get_center_xyxy()[1]),
            linestyle="-",
            color="blue",
        )  # center min line

        plt.plot(
            (center.get_center_xyxy()[0], head.get_center_xyxy()[0]),
            (center.get_center_xyxy()[1], head.get_center_xyxy()[1]),
            linestyle="-",
            color="mediumseagreen",
        )  # center head line

        plt.plot(
            (center.get_center_xyxy()[0], max.get_center_xyxy()[0]),
            (center.get_center_xyxy()[1], max.get_center_xyxy()[1]),
            linestyle="-",
            color="fuchsia",
        )  # center max line

        # plot a vector
        print(f"vectors: {vectors}")
        for v in vectors:
            print(f"v is :{v}")
            plt.plot((v[0], v[2]), (v[1], v[3]), linestyle="-", color="red")

        plt.title(f"{gauge.xyxy}")

        # ---------------------------------------Draw a circle
        # circle = plt.gca().add_patch(
        #     patches.Circle(
        #         (clock_center),clock_r, color="gold"
        #     )
        # )

        plt.show()
