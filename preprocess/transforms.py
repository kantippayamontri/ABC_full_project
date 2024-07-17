from icecream import ic
import numpy as np
from utils import Utils, Constants
import random
import cv2


class Transform:

    def __init__(self, img_path, bb_path):
        self.img_path = img_path
        self.bb_path = bb_path

    def get_img_path(
        self,
    ):
        return self.img_path

    def get_bb_path(
        self,
    ):
        return self.bb_path

    def get_img_bb_path(
        self,
    ):
        return (self.img_path, self.bb_path)

    def prepare_input(self, img, bb):
        # format for albumentation yolo = [cx,cy,w,h,class]
        permutation = [1, 2, 3, 4, 0]
        bb = bb[:, permutation]
        return img, bb

    def prepare_output(self, img, bb):
        # prepare output for bb
        if len(bb) != 0:
            inversePermutation = [4, 0, 1, 2, 3]
            bb = bb[:, inversePermutation]

        return img, bb

    def save_img(self, img, path):
        Utils.save_image(img=img, filepath=path)

    def save_bb(self, bb_list, path):
        Utils.save_bb(bb_list=bb_list, bb_path=path)

    def make_new_name(self, name, function_name, prefix):
        file_name, file_prefix = str(name).split(".")

        return f"{file_name}_{function_name}_{prefix}.{file_prefix}"

    def get_output_tramsformed(self, transformed):
        # bboxes = x,y,w,h,c
        _bb = np.array([list(bb_temp) for bb_temp in transformed["bboxes"]])
        for j in range(len(_bb)):
            for i in range(len(_bb[j]) - 1):
                if _bb[j][i] < 0.0:
                    _bb[j][i] = 0.0

                if _bb[j][i] > 1.0:
                    _bb[j][i] = 1.0

        return transformed["image"], _bb

    def crop(
        self,
        img,
        bb,
        class_crop_list,
        need_resize,
        target_size,
        class_ignore,
        pixel_added,
        save_path,
    ):

        full_img = img.copy()
        bb_temp = bb.copy()

        for class_crop in class_crop_list:
            # ic(f"class crop : {class_crop}, class crp list: {class_crop_list}")
            # ic(f"class crop : {class_crop}")
            bb = []  # for store bb use after crop
            bb_crop = []  # for store bb that use to crop the coordinate

            for _bb in bb_temp:
                if int(_bb[-1]) == int(class_crop):
                    bb_crop.append(_bb)
                    bb.append(_bb)  # also add class crop into bb
                elif _bb[-1] not in class_ignore:
                    bb.append(_bb)

            bb = np.array(bb)
            bb_crop = np.array(bb_crop)

            if len(bb_crop) == 0:
                return img, bb

            for index_crop, _bbc in enumerate(
                bb_crop
            ):  # loop image to get new image that crop from _bbc
                bb_use = []
                for _bb in bb:  # loop check which bb inside the _bbc
                    _, bb1 = self.prepare_output(
                        img=None, bb=np.array([_bbc])
                    )  # for bb crop
                    box1 = Utils.change_format_yolo2xyxy(
                        img_size=full_img.shape, bb=bb1[0], with_class=True
                    )["bb"]

                    __, bb2 = self.prepare_output(
                        img=None, bb=np.array([_bb])
                    )  # for bb use
                    box2 = Utils.change_format_yolo2xyxy(
                        img_size=full_img.shape, bb=bb2[0], with_class=True
                    )["bb"]

                    if Utils.calculate_intersection(box1=box1, box2=box2):
                        bb_use.append(_bb.copy())

                if len(bb_use) == 0:
                    continue

                bb_use = np.array(bb_use)

                _, _bbc = self.prepare_output(
                    img=None, bb=np.array([_bbc])
                )  # _bbc need this format [c,x,y,w,h] -> use change format from yolo to xyxy
                _bbc = _bbc[0]

                albu_transform = Utils.albu_crop_img_bb(
                    img=full_img,
                    bb_crop=_bbc,
                    format=Constants.BoundingBoxFormat.YOLOV8,
                    add_pixels=random.randint(0, pixel_added),
                    with_class=True,
                )

                transformed = albu_transform(image=full_img, bboxes=bb_use)
                new_img, new_bb = self.get_output_tramsformed(transformed=transformed)

                if need_resize:
                    new_img, new_bb = self.resize(
                        img=new_img,
                        bb=new_bb,
                        target_size=target_size,
                        format=Constants.BoundingBoxFormat.YOLOV8,
                    )
                    new_bb = np.array([list(bb_temp) for bb_temp in new_bb])

                # new name for crop
                new_name_img = self.make_new_name(
                    name=self.img_path.name,
                    function_name="crop",
                    prefix=f"{class_crop}_{index_crop}",
                )
                new_name_bb = self.make_new_name(
                    name=self.bb_path.name,
                    function_name="crop",
                    prefix=f"{class_crop}_{index_crop}",
                )

                # save crop image
                self.save_img(
                    img=new_img, path=save_path / Constants.image_folder / new_name_img
                )
                # save crop bb
                _, new_bb_save = self.prepare_output(img=None, bb=new_bb)
                self.save_bb(
                    bb_list=new_bb_save,
                    path=save_path / Constants.label_folder / new_name_bb,
                )

        return img, bb

    def resize(self, img, bb, target_size, format=None):
        albu_transform = Utils.albu_resize_img_bb(
            target_size=target_size, format=format
        )
        transformed = albu_transform(image=img, bboxes=bb)

        new_img, new_bb = self.get_output_tramsformed(transformed=transformed)

        return new_img, new_bb

    def gray(self, img, bb, format=None, p=1.0):
        albu_transform = Utils.albu_grayscale(format=format, p=p)
        transformed = albu_transform(image=img, bboxes=bb)
        new_img, new_bb = self.get_output_tramsformed(transformed=transformed)
        return new_img, new_bb

    def channel_shuffle(self, img, bb, format=None, p=1.0):
        albu_transform = Utils.albu_channelshuffle(format=format, p=p)
        transformed = albu_transform(image=img, bboxes=bb)
        new_img, new_bb = self.get_output_tramsformed(transformed=transformed)
        return new_img, new_bb

    def multiplicative_noise(
        self, img, bb, format=None, p=1.0, multiplier=[0.0, 1.0], element_wise=True
    ):
        albu_transform = Utils.albu_multiplicative_noise(
            format=format, multiplier=multiplier, element_wise=element_wise, p=p
        )
        transformed = albu_transform(image=img, bboxes=bb)
        new_img, new_bb = self.get_output_tramsformed(transformed=transformed)
        return new_img, new_bb

    def blur(self, img, bb, format=None, p=1.0, blur_limit=[7, 7]):
        albu_transform = Utils.albu_blur(blur_limit=blur_limit, p=p, format=format)
        transformed = albu_transform(image=img, bboxes=bb)
        new_img, new_bb = self.get_output_tramsformed(transformed=transformed)
        return new_img, new_bb

    def rotate(self, img, bb, format=None, p=1.0, limit=[-10, 10]):
        albu_transform = Utils.albu_rotate(format=format, p=p, limit=limit)
        transformed = albu_transform(image=img, bboxes=bb)
        new_img, new_bb = self.get_output_tramsformed(transformed=transformed)
        return new_img, new_bb

    def color_jitter(
        self,
        img,
        bb,
        format=None,
        p=1.0,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2,
    ):
        albu_transform = Utils.albu_color_jitter(
            format=format,
            p=p,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        transformed = albu_transform(image=img, bboxes=bb)
        new_img, new_bb = self.get_output_tramsformed(transformed=transformed)
        return new_img, new_bb

    def longest_max_size(self, img, bb, format=None, p=1.0, max_size=640):
        albu_transform = Utils.albu_longest_max_size(
            format=format, p=p, max_size=max_size
        )
        transformed = albu_transform(image=img, bboxes=bb)
        new_img, new_bb = self.get_output_tramsformed(transformed=transformed)
        return new_img, new_bb

    def pad_if_needed(self, img, bb, format=None, p=1.0, min_width=640, min_height=640):
        albu_transform = Utils.albu_pad_if_needed(
            format=format, p=p, min_width=min_width, min_height=min_height
        )
        transformed = albu_transform(image=img, bboxes=bb)
        new_img, new_bb = self.get_output_tramsformed(transformed=transformed)
        return new_img, new_bb

    def gray_erosion_dilate(
        self,
        img,
        bb,
        format=None,
        p=1.0,
    ):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dark_threshold = 100
        bright_threshold = 150

        check_image_dark = self.is_image_dark(image=img, threshold=dark_threshold)
        check_image_bright = self.is_image_bright(image=img, threshold=bright_threshold)

        # tag="same birghtnesss"
        if check_image_dark:
            # tag = "increase brightness"
            if self.mean_gray_img(image=img) <= dark_threshold - 20:
                img = self.increase_brightness(img=img, value=50 + 20)
            else:
                img = self.increase_brightness(img=img, value=50)
        elif check_image_bright:
            ...

        # create single channel img
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # make threshold image
        res = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5
        )

        # for image erosion
        erosion_kernel_before = np.ones((2, 2), np.uint8)
        erosion_img_before = cv2.erode(res, kernel=erosion_kernel_before, iterations=1)

        # for image dilation
        dilate_kernel = np.ones((3, 3), np.uint8)
        dilate_img = cv2.dilate(erosion_img_before, kernel=dilate_kernel, iterations=1)

        # for image erosion
        erosion_kernel_after = np.ones((2, 2), np.uint8)
        erosion_img_after = cv2.erode(
            dilate_img, kernel=erosion_kernel_after, iterations=1
        )

        # # Find the edges in the image using canny detector
        # edges = cv2.Canny(dilate_img, 130, 255)

        # for make binary image from 1 channel to 3 channel
        three_channel_image = cv2.cvtColor(erosion_img_after, cv2.COLOR_GRAY2BGR)

        return three_channel_image, bb

    def clock_pre_min_head(self, img=None, bb=[]):
        import torch
        import torchvision.ops.boxes as bops

        """
        bb = x y w h c
        """

        min_class = Constants.map_data_dict["clock"]["target"].index("min")
        head_class = Constants.map_data_dict["clock"]["target"].index("head")

        min_bb = []
        head_bb = []

        new_bb = []

        for index, _bb in enumerate(bb):
            if _bb[4] == min_class:
                min_bb.append(
                    {"index": index, "bb": [_bb[4], _bb[0], _bb[1], _bb[2], _bb[3]]}
                )
            elif _bb[4] == head_class:
                head_bb.append(
                    {"index": index, "bb": [_bb[4], _bb[0], _bb[1], _bb[2], _bb[3]]}
                )
                new_bb.append(bb[index])
            else:
                new_bb.append(bb[index])

        if len(min_bb) == 0 or len(head_bb) == 0:
            return img, bb

        # is_overlap = False

        # loop head
        for _head_dict in head_bb:
            # loop min
            for _min_dict in min_bb:
                _head_xyxy = Utils.change_format_yolo2xyxy(
                    img_size=img.shape, bb=_head_dict["bb"], with_class=True
                )["bb"]
                _min_xyxy = Utils.change_format_yolo2xyxy(
                    img_size=img.shape, bb=_min_dict["bb"], with_class=True
                )["bb"]

                # check percent overlap
                _head_xyxy = torch.tensor(
                    [
                        [
                            _head_xyxy[0][0],
                            _head_xyxy[0][1],
                            _head_xyxy[1][0],
                            _head_xyxy[1][1],
                        ]
                    ],
                    dtype=torch.float,
                )
                _min_xyxy = torch.tensor(
                    [
                        [
                            _min_xyxy[0][0],
                            _min_xyxy[0][1],
                            _min_xyxy[1][0],
                            _min_xyxy[1][1],
                        ]
                    ],
                    dtype=torch.float,
                )

                iou = bops.box_iou(_min_xyxy, _head_xyxy)

                if iou >= 0.3:
                    ...
                    # is_overlap = True
                    # Utils.visualize_img_bb(img=img, bb=np.array([ [b[4], b[0], b[1], b[2], b[3]] for b in bb]), with_class=True, labels=["gauge", "min", "max", "center", "head", "bottom"])
                else:
                    new_bb.append(bb[_min_dict["index"]])

        # if is_overlap:
        #     ic(f"is_overlap : True")
        #     ic(new_bb)
        #     Utils.visualize_img_bb(img=img, bb=np.array([ [b[4], b[0], b[1], b[2], b[3]] for b in new_bb]), with_class=True, labels=["gauge", "min", "max", "center", "head", "bottom"])

        return img, np.array(new_bb)

    def clock_pre_min_max(
        self, img=None, bb=[]
    ):  # when have multiple min -> use the farest min
        """
        bb = x y w h
        """
        min_class = Constants.map_data_dict["clock"]["target"].index("min")
        max_class = Constants.map_data_dict["clock"]["target"].index("max")
        gauge_class = Constants.map_data_dict["clock"]["target"].index("gauge")

        min_bb = []
        max_bb = []
        gauge_bb = []

        new_bb = []

        for index, _bb in enumerate(bb):
            if _bb[4] == min_class:
                min_bb.append(
                    {"index": index, "bb": [_bb[4], _bb[0], _bb[1], _bb[2], _bb[3]]}
                )
            elif _bb[4] == max_class:
                max_bb.append(
                    {"index": index, "bb": [_bb[4], _bb[0], _bb[1], _bb[2], _bb[3]]}
                )
            elif _bb[4] == gauge_class:
                gauge_bb.append(
                    {"index": index, "bb": [_bb[4], _bb[0], _bb[1], _bb[2], _bb[3]]}
                )
                new_bb.append(_bb)
            else:
                new_bb.append(_bb)

        if len(gauge_bb) == 1:  # gauge that don't have other gauge inside

            if len(min_bb) > 1:
                min_xyxy_list = [
                    Utils.change_format_yolo2xyxy(
                        img_size=img.shape, bb=_bb["bb"], with_class=True
                    )["bb"]
                    for _bb in min_bb
                ]
                min_xyxy_list = [
                    [_bb[0][0], _bb[0][1], _bb[1][0], _bb[1][1]]
                    for _bb in min_xyxy_list
                ]
                min_xyxy_list = np.array(min_xyxy_list)

                average_min = np.sum(min_xyxy_list, axis=0) / len(min_xyxy_list)
                average_min = average_min.astype(np.int64)

                average_min_yolo = Utils.change_format_xyxy2yolo(
                    img_size=img.shape, bb=average_min, cls=min_class, normalize=True
                )
                average_min_yolo = average_min_yolo["bb"]
                average_min_yolo.append(min_class)
                new_bb.append(average_min_yolo)
            elif len(min_bb) == 1:
                temp = min_bb[0]["bb"]
                new_bb.append(
                    [
                        temp[1],
                        temp[2],
                        temp[3],
                        temp[4],
                        temp[0],
                    ]
                )

            if len(max_bb) > 1:
                max_xyxy_list = [
                    Utils.change_format_yolo2xyxy(
                        img_size=img.shape, bb=_bb["bb"], with_class=True
                    )["bb"]
                    for _bb in max_bb
                ]
                max_xyxy_list = [
                    [_bb[0][0], _bb[0][1], _bb[1][0], _bb[1][1]]
                    for _bb in max_xyxy_list
                ]
                max_xyxy_list = np.array(max_xyxy_list)

                average_max = np.sum(max_xyxy_list, axis=0) / len(max_xyxy_list)
                average_max = average_max.astype(np.int64)

                average_max_yolo = Utils.change_format_xyxy2yolo(
                    img_size=img.shape, bb=average_max, cls=max_class, normalize=True
                )
                average_max_yolo = average_max_yolo["bb"]
                average_max_yolo.append(max_class)
                new_bb.append(average_max_yolo)

            elif len(max_bb) == 1:
                temp = max_bb[0]["bb"]
                new_bb.append([temp[1], temp[2], temp[3], temp[4], temp[0]])

        return img, np.array(new_bb)

    def clock_only_one_gauge(self, img, bb):  # use only img that have only one gauge
        """
        bb = x y w h
        """
        gauge_class = Constants.map_data_dict["clock"]["target"].index("gauge")
        # count number of gauge
        count_gauge = 0
        for index, _bb in enumerate(bb):
            if _bb[4] == gauge_class:
                count_gauge += 1

        if count_gauge > 1:
            Utils.visualize_img_bb(img=img, bb=np.array([ [b[4], b[0], b[1], b[2], b[3]] for b in bb]), with_class=True, labels=["gauge", "min", "max", "center", "head", "bottom"])
            return None, None

        return img, bb

    def clock_full_only(self, img, bb):  # choose image only have min, max ,center ,head
        if img is None or bb is None:
            return None, None

        gauge_class = Constants.map_data_dict["clock"]["target"].index("gauge")
        min_class = Constants.map_data_dict["clock"]["target"].index("min")
        max_class = Constants.map_data_dict["clock"]["target"].index("max")
        head_class = Constants.map_data_dict["clock"]["target"].index("head")
        center_class = Constants.map_data_dict["clock"]["target"].index("center")

        check_gauge = False
        check_min = False
        check_max = False
        check_head = False
        check_center = False

        for _bb in bb:
            if int(_bb[4]) == gauge_class:
                check_gauge = True
            elif int(_bb[4]) == min_class:
                check_min = True
            elif int(_bb[4]) == max_class:
                check_max = True
            elif int(_bb[4]) == head_class:
                check_head = True
            elif int(_bb[4]) == center_class:
                check_center = True

        if check_gauge and check_min and check_max and check_head and check_center:
            # Utils.visualize_img_bb(img=img, bb=np.array([ [b[4], b[0], b[1], b[2], b[3]] for b in bb]), with_class=True, labels=["gauge", "min", "max", "center", "head", "bottom"])
            return img, bb

        return None, None

    def add_needle(self, img, bb):
        if (img is None) or (bb is None):
            return None, None

        head_class = Constants.map_data_dict["clock"]["target"].index("head")
        center_class = Constants.map_data_dict["clock"]["target"].index("center")
        bottom_class = Constants.map_data_dict["clock"]["target"].index("bottom")

        check_head = False
        check_center = False
        check_bottom = False

        center_list = []
        head_list = []
        bottom_list = []

        new_bb = [] 

        for _bb in bb:
            if int(_bb[4]) == head_class:
                check_head = True
                head_list.append(_bb)
            elif int(_bb[4]) == center_class:
                check_center = True
                center_list.append(_bb)
            elif int(_bb[4]) == bottom_class:
                check_bottom = True
                bottom_list.append(_bb)
            
            new_bb.append(_bb)

        if check_head and check_center and check_bottom:
            # found head center and bottom -> add needle from head and bottom
            # _center_bb = center_list[0] # use center at index 0
            _bottom_bb = bottom_list[0]
            for _head_bb in head_list:
                needle_yolo = self.create_needle(
                    bb_one=_head_bb,
                    bb_two=_bottom_bb,
                    img_size=img.shape,
                    to_yolo=True, # convert to yolo format  
                    cls=6
                )
                # needle_yolo = needle_yolo["bb"].append(needle_yolo["class"])
                needle_yolo = needle_yolo["bb"]
                needle_yolo.append(6) # needle class = 6 
                needle_yolo = np.array(needle_yolo)
                new_bb.append(needle_yolo)
            # Utils.visualize_img_bb(img=img, bb=np.array([ [b[4], b[0], b[1], b[2], b[3]] for b in new_bb]), with_class=True, labels=["gauge", "min", "max", "center", "head", "bottom", "needle"])
            return img, np.array(new_bb)

        elif check_head and check_center:
            # found head and center -> add needle for head and center
            _center_bb = center_list[0]
            for _head_bb in head_list:
                needle_yolo = self.create_needle(
                    bb_one=_head_bb,
                    bb_two=_center_bb,
                    img_size=img.shape,
                    to_yolo=True, # convert to yolo format  
                    cls=6
                )
                # needle_yolo = needle_yolo["bb"].append(needle_yolo["class"])
                needle_yolo = needle_yolo["bb"]
                needle_yolo.append(6) # needle class = 6 
                needle_yolo = np.array(needle_yolo)
                new_bb.append(needle_yolo)
            # Utils.visualize_img_bb(img=img, bb=np.array([ [b[4], b[0], b[1], b[2], b[3]] for b in new_bb]), with_class=True, labels=["gauge", "min", "max", "center", "head", "bottom", "needle"])
            return img, np.array(new_bb)
        else:
            return img, bb


    def create_needle(self, bb_one, bb_two, img_size, to_yolo=False, cls=0):
        """
        bb = x y w h c
        """
        bb_one_xyxy = Utils.change_format_yolo2xyxy(
            img_size=img_size,
            bb=[bb_one[4], bb_one[0], bb_one[1], bb_one[2], bb_one[3]],
            with_class=True,
        )["bb"]

        bb_two_xyxy = Utils.change_format_yolo2xyxy(
            img_size=img_size,
            bb=[bb_two[4], bb_two[0], bb_two[1], bb_two[2], bb_two[3]],
            with_class=True,
        )["bb"]

        center_one_x, center_one_y = self.get_center_xyxy(bb=bb_one_xyxy)
        center_two_x, center_two_y = self.get_center_xyxy(bb=bb_two_xyxy)

        needle_xyxy = [
            int(min(center_one_x, center_two_x)), int(min(center_one_y, center_two_y)),
            int(max(center_one_x, center_two_x)), int(max(center_one_y, center_two_y)),
        ]

        if to_yolo: # convert to yolo format
            needle_yolo = Utils.change_format_xyxy2yolo(
                img_size=img_size,
                bb=needle_xyxy,
                cls=cls,
                normalize=True
            )
            return needle_yolo

        return needle_xyxy

    def get_center_xyxy(self, bb):
        """
        bb = [(x,y), (x,y)]
        """
        x_min, y_min = bb[0]
        x_max, y_max = bb[1]

        return [x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2]

    def resize_bb(self, img, bb, cls, percent=10):
        """
        bb = x,y,w,h,c
        """
        new_bb = []

        for _bb in bb:
            if int(_bb[4]) not in cls:
                new_bb.append(_bb)

        for _cls in cls:  # loop each class need to resize
            for _bb in bb:  # loop each bounding box in class
                if (
                    int(_bb[4]) == _cls
                ):  # bounding box in that class that need to resize
                    _bb_xyxy = Utils.change_format_yolo2xyxy(
                        img_size=img.shape,
                        bb=[_bb[4], _bb[0], _bb[1], _bb[2], _bb[3]],
                        with_class=True,
                    )["bb"]

                    x_min, y_min = _bb_xyxy[0]
                    x_max, y_max = _bb_xyxy[1]

                    w = x_max - x_min
                    h = y_max - y_min

                    w_add = w * (percent / 100) / 2
                    h_add = h * (percent / 100) / 2

                    x_min = x_min - w_add
                    y_min = y_min - h_add

                    x_max = x_max + w_add
                    y_max = y_max + h_add

                    resize_bb = np.array([x_min, y_min, x_max, y_max])
                    resize_bb_yolo = Utils.change_format_xyxy2yolo(
                        img_size=img.shape, bb=resize_bb, cls=_cls, normalize=True
                    )
                    resize_bb_yolo = resize_bb_yolo["bb"]
                    resize_bb_yolo.append(_cls)
                    new_bb.append(resize_bb_yolo)

        return img, np.array(new_bb)

    def mean_gray_img(self, image):  # input is color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return mean_brightness

    def is_image_dark(self, image, threshold=100):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return mean_brightness < threshold

    def is_image_bright(self, image, threshold=100):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return mean_brightness > threshold

    def decrease_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] -= value
        v[v <= value] = 0

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def increase_brightness(self, img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def transform_dict_function(
        self,
        function_name,
        function_parameter,
        img,
        bb,
        target_folder_path,
        dataset_type,
    ):

        img, bb = self.prepare_input(img=img, bb=bb)

        # ic(function_name, function_parameter)
        if function_name == "CROP":
            img, bb = self.crop(
                img=img.copy(),
                bb=bb.copy(),
                class_crop_list=function_parameter["CLASS_CROP_LIST"],
                need_resize=function_parameter["NEED_RESIZE"],
                target_size=(
                    function_parameter["TARGET_WIDTH"],
                    function_parameter["TARGET_HEIGHT"],
                ),
                class_ignore=function_parameter["CLASS_IGNORE"],
                pixel_added=function_parameter["ADD_PIXEL"],
                save_path=target_folder_path,
            )
        elif function_name == "RESIZE":
            img, bb = self.resize(
                img=img.copy(),
                bb=bb.copy(),
                target_size=(
                    function_parameter["TARGET_WIDTH"],
                    function_parameter["TARGET_HEIGHT"],
                ),
                format=Constants.BoundingBoxFormat.YOLOV8,
            )
        elif function_name == "GRAY":
            img, bb = self.gray(
                img=img.copy(),
                bb=bb.copy(),
                format=Constants.BoundingBoxFormat.YOLOV8,
                p=float(function_parameter["P"]),
            )
        elif function_name == "CHANNEL_SHUFFLE":
            img, bb = self.channel_shuffle(
                img=img.copy(),
                bb=bb.copy(),
                format=Constants.BoundingBoxFormat.YOLOV8,
                p=float(function_parameter["P"]),
            )
        elif function_name == "MULTIPLICATIVE_NOISE":
            img, bb = self.multiplicative_noise(
                img=img.copy(),
                bb=bb.copy(),
                format=Constants.BoundingBoxFormat.YOLOV8,
                p=function_parameter["P"],
                multiplier=function_parameter["MULTIPLIER"],
                element_wise=function_parameter["ELEMENT_WISE"],
            )
        elif function_name == "BLUR":
            img, bb = self.blur(
                img=img.copy(),
                bb=bb.copy(),
                format=Constants.BoundingBoxFormat.YOLOV8,
                p=function_parameter["P"],
                blur_limit=function_parameter["BLUR_LIMIT"],
            )
        elif function_name == "ROTATE":
            img, bb = self.rotate(
                img=img.copy(),
                bb=bb.copy(),
                format=Constants.BoundingBoxFormat.YOLOV8,
                p=function_parameter["P"],
                limit=function_parameter["LIMIT"],
            )
        elif function_name == "COLOR_JITTER":
            img, bb = self.color_jitter(
                img=img.copy(),
                bb=bb.copy(),
                format=Constants.BoundingBoxFormat.YOLOV8,
                p=function_parameter["P"],
                brightness=function_parameter["BRIGHTNESS"],
                contrast=function_parameter["CONTRAST"],
                saturation=function_parameter["SATURATION"],
                hue=function_parameter["HUE"],
            )
        elif function_name == "LONGEST_MAX_SIZE":
            img, bb = self.longest_max_size(
                img=img.copy(),
                bb=bb.copy(),
                format=Constants.BoundingBoxFormat.YOLOV8,
                p=function_parameter["P"],
                max_size=function_parameter["MAX_SIZE"],
            )
        elif function_name == "PAD_IF_NEEDED":
            img, bb = self.pad_if_needed(
                img=img.copy(),
                bb=bb.copy(),
                format=Constants.BoundingBoxFormat.YOLOV8,
                p=function_parameter["P"],
                min_width=function_parameter["MIN_WIDTH"],
                min_height=function_parameter["MIN_HEIGHT"],
            )
        elif function_name == "GRAY_EROSION_DILATE":
            img, bb = self.gray_erosion_dilate(
                img=img.copy(),
                bb=bb.copy(),
                format=Constants.BoundingBoxFormat.YOLOV8,
                p=function_parameter["P"],
            )
        elif function_name == "CLOCK":
            if "PREPROCESS_MIN_HEAD" in function_parameter.keys():
                if function_parameter["PREPROCESS_MIN_HEAD"]:
                    # ic(f"---> function: PREPROCESS_MIN_HEAD")
                    img, bb = self.clock_pre_min_head(img=img.copy(), bb=bb.copy())

            if "PREPROCESS_MIN_MAX" in function_parameter.keys():
                if function_parameter['PREPROCESS_MIN_MAX']:
                    # ic(f"---> function: PREPROCESS_MIN_MAX")
                    img, bb = self.clock_pre_min_max(img=img.copy(), bb=bb.copy())

            if "PREPROCESS_ONLY_ONE_GAUGE" in function_parameter.keys():
                if function_parameter["PREPROCESS_ONLY_ONE_GAUGE"]:
                    # ic(f"---> function: PREPROCESS_ONLY_ONE_GAUGE")
                    img, bb = self.clock_only_one_gauge(img=img.copy(), bb=bb.copy())

            if "PREPROCESS_FULL_CLASS" in function_parameter.keys():
                if function_parameter["PREPROCESS_FULL_CLASS"]:
                    # ic(f"---> function: PREPROCESS_FULL_CLASS")
                    if (img is not None) and (bb is not None):
                        img, bb = self.clock_full_only(img=img.copy(), bb=bb.copy())
                    else:  
                        print("can not full class")

            if "ADD_NEEDLE" in  function_parameter.keys():
                if function_parameter["ADD_NEEDLE"]:
                    if (img is not None) and (bb is not None):
                        img, bb = self.add_needle(img=img.copy(), bb=bb.copy())
                    else:
                        print(f"can not add_needle img is None or bb in None")

        elif function_name == "RESIZE_BB":
            img, bb = self.resize_bb(
                img=img.copy(),
                bb=bb.copy(),
                cls=function_parameter["CLASS"],  # list of int
                percent=function_parameter["PERCENT"],
            )

        else:
            print(f"\t\t Augment function {function_name} is not found.")
            img, bb = self.prepare_output(img=img, bb=bb)
            return img, bb

        # for some function need to replace original image
        try:
            img, bb = self.prepare_output(img=img, bb=bb)
            if function_parameter["REPLACE"] and (img is not None) and (bb is not None):
                self.save_img(img=img, path=self.img_path)
                self.save_bb(bb_list=bb, path=self.bb_path)
        except Exception as e :
            print(f"ERROR: {e}")
            print(bb)
            Utils.deleted_file(file_path=self.img_path)
            Utils.deleted_file(file_path=self.bb_path)

        return img, bb
