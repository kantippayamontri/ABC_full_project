from icecream import ic
from inference.inference_constants import InferenceConstants
from inference.inference_utils import InferenceUtils
from enum import Enum, auto


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

        self.gaugeList = self.makeGaugesList()
        self.displayList = self.makeDisplayList()
        self.frameList = self.makeFrameList()

        self.nGauges = len(self.gaugeList)
        self.nDisplays = len(self.displayList)
        self.nFrames = len(self.frameList)

        self.frameInGauge = self.frameInGauge()
        self.frameInDisplay = self.frameInDisplay()
        # for box in self.boxes_list:
        #     box.printBox()

    class COORDINATES(Enum):
        X1 = auto()
        Y1 = auto()
        X2 = auto()
        Y2 = auto()

    def getCoordinatesRealImage(
        self, ori_shape, want_shape, box_coor
    ):  # convert cooridates from convert_shape to ori_shape
        ic(ori_shape, want_shape, box_coor)
        ori_w = ori_shape[1]
        ori_h = ori_shape[0]

        want_w = want_shape[1]
        want_h = want_shape[0]

        box_c = {
            self.COORDINATES.X1: box_coor[0],
            self.COORDINATES.Y1: box_coor[1],
            self.COORDINATES.X2: box_coor[2],
            self.COORDINATES.Y2: box_coor[3],
        }

        new_x1 = self.NewRangRatio(
            ori_range=ori_w, new_range=want_w, ori_value=box_c[self.COORDINATES.X1]
        )
        new_x2 = self.NewRangRatio(
            ori_range=ori_w, new_range=want_w, ori_value=box_c[self.COORDINATES.X2]
        )
        new_y1 = self.NewRangRatio(
            ori_range=ori_h, new_range=want_h, ori_value=box_c[self.COORDINATES.Y1]
        )
        new_y2 = self.NewRangRatio(
            ori_range=ori_h, new_range=want_h, ori_value=box_c[self.COORDINATES.Y2]
        )

       # ic(f"ori_x1: {box_c[self.COORDINATES.X1]}, new_x1: {new_x1}")
       # ic(f"ori_x2: {box_c[self.COORDINATES.X2]}, new_x1: {new_x2}")
       # ic(f"ori_y2: {box_c[self.COORDINATES.Y1]}, new_x1: {new_y1}")
       # ic(f"ori_x1: {box_c[self.COORDINATES.Y2]}, new_x1: {new_y2}")

        return [new_x1, new_y1, new_x2, new_y2]

    def NewRangRatio(self, ori_range, new_range, ori_value):
        return int((ori_value * new_range) / ori_range)

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
    ):
        self.box = box
        self.cls = cls
        self.conf = conf
        self.data = data
        self.xywh = xywh
        self.xywhn = xywhn
        self.xyxy = xyxy
        self.xyxyn = xyxyn

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
        print()
