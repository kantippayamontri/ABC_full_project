from icecream import ic
import numpy as np


class Transform:
    # transform_dict = {
    #     "CROP" : crop,
    # }

    def __init__(
        self,
    ):
        return

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

    def crop(
        self, img, bb, class_crop, need_resize, target_size, class_ignore, pixel_added
    ): 
        ic(class_crop, need_resize, target_size, class_ignore, pixel_added)
        ic(bb)

        bb_crop = [_bb for _bb in bb if int(_bb[-1]) == int(class_crop)]
        

    def transform_dict_function(self, function_name, function_parameter, img, bb):
        img, bb = self.prepare_input(img=img, bb=bb)

        ic(function_name, function_parameter)
        if function_name == "CROP":
            self.crop(
                img=img,
                bb=bb,
                class_crop=function_parameter["CLASS_CROP"],
                need_resize=function_parameter["NEED_RESIZE"],
                target_size=(
                    function_parameter["TARGET_WIDTH"],
                    function_parameter["TARGET_HEIGHT"],
                ),
                class_ignore=function_parameter["CLASS_IGNORE"],
                pixel_added=function_parameter["ADD_PIXEL"],
            )

        img, bb = self.prepare_output(img=img, bb=bb)
        return img, bb
