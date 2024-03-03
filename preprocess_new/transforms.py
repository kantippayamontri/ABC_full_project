from icecream import ic
import numpy as np
from utils import Utils, Constants
import random


class Transform:

    def __init__(self, img_path, bb_path):
        self.img_path = img_path
        self.bb_path = bb_path

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

    def new_name_pre(self, name, function_name, prefix):
        file_name, file_prefix = str(name).split(".")

        return f"{file_name}_{function_name}_{prefix}.{file_prefix}"

    def get_output_tramsformed(self, transformed):
        return transformed["image"], np.array(
            [list(bb_temp) for bb_temp in transformed["bboxes"]]
        )

    def crop(
        self,
        img,
        bb,
        class_crop,
        need_resize,
        target_size,
        class_ignore,
        pixel_added,
        save_path,
    ):

        full_img = img.copy()
        bb_temp = bb.copy()

        bb = []  # for store bb use after crop
        bb_crop = []  # for store bb that use to crop the coordinate

        for _bb in bb_temp:
            if _bb[-1] == class_crop:
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
            new_name_img = self.new_name_pre(
                name=self.img_path.name,
                function_name="crop",
                prefix=f"{class_crop}_{index_crop}",
            )
            new_name_bb = self.new_name_pre(
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
        albu_transform = Utils.albu_grayscale(format=format,p=1.0)
        transformed = albu_transform(image=img, bboxes=bb)
        new_img, new_bb = self.get_output_tramsformed(transformed=transformed)
        return new_img, new_bb

    def transform_dict_function(
        self, function_name, function_parameter, img, bb, target_folder_path
    ):
        img, bb = self.prepare_input(img=img, bb=bb)

        # ic(function_name, function_parameter)
        if function_name == "CROP":
            img, bb = self.crop(
                img=img.copy(),
                bb=bb.copy(),
                class_crop=function_parameter["CLASS_CROP"],
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
            img, bb = self.gray(img=img.copy(), bb=bb.copy(), format=Constants.BoundingBoxFormat.YOLOV8)
        else:
            ...

        img, bb = self.prepare_output(img=img, bb=bb)

        # for some function need to replace original image
        if function_parameter["REPLACE"]:
            self.save_img(img=img, path=self.img_path)
            self.save_bb(bb_list=bb, path=self.bb_path)

        return img, bb