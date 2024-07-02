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
        _bb = np.array(
            [list(bb_temp) for bb_temp in transformed["bboxes"]]
        )
        for j in range(len(_bb)):
            for i in range(len(_bb[j])-1):
                if _bb[j][i] < 0.0:
                    _bb[j][i]  =0.0
                
                if _bb[j][i] > 1.0:
                    _bb[j][i] =1.0
            
            
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
        ic(f"gray img dim before: {img.shape}")
        albu_transform = Utils.albu_grayscale(format=format, p=1.0)
        transformed = albu_transform(image=img, bboxes=bb)
        new_img, new_bb = self.get_output_tramsformed(transformed=transformed)
        ic(f"gray img dim after: {new_img.shape}")
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
        albu_transform = Utils.albu_pad_if_needed(format=format, p=p, min_width=min_width, min_height=min_height)
        transformed = albu_transform(image=img, bboxes=bb)
        new_img, new_bb = self.get_output_tramsformed(transformed=transformed)
        return new_img, new_bb
    
    def gray_erosion_dilate(self, img, bb, format=None, p=1.0, ):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow(f"mean image : {self.mean_gray_img(image=img)}", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        dark_threshold = 100
        bright_threshold = 150

        check_image_dark =self.is_image_dark(image=img, threshold=dark_threshold) 
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
            # tag = "decrease brightness"
            # img = self.decrease_brightness(img=img, value=50)
        
        # cv2.imshow(tag, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # create single channel img 
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        #make threshold image
        res = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)

        #for image erosion
        erosion_kernel_before = np.ones((2,2), np.uint8)
        erosion_img_before = cv2.erode(res, kernel=erosion_kernel_before, iterations=1)

        #for image dilation
        dilate_kernel = np.ones((3,3), np.uint8)
        dilate_img = cv2.dilate(erosion_img_before, kernel=dilate_kernel, iterations=1)

        #for image erosion
        erosion_kernel_after = np.ones((2,2), np.uint8)
        erosion_img_after = cv2.erode(dilate_img, kernel=erosion_kernel_after, iterations=1)

        # # Find the edges in the image using canny detector
        # edges = cv2.Canny(dilate_img, 130, 255)

        # for make binary image from 1 channel to 3 channel
        three_channel_image = cv2.cvtColor(erosion_img_after, cv2.COLOR_GRAY2BGR)

        # Display the images
        # cv2.imshow("Gray Image", gray_image)
        # # cv2.imshow("Equalize Image", equ_image)
        # cv2.imshow("Binary Image", dilate_img)
        # cv2.imshow("Three-Channel Image", three_channel_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return three_channel_image, bb
    
    def mean_gray_img(self,image): #input is color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return mean_brightness
        
    
    def is_image_dark(self,image, threshold=100):
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
        v[v > lim ] -= value
        v[v <= value] = 0

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
        
    
    def increase_brightness(self,img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def transform_dict_function(
        self, function_name, function_parameter, img, bb, target_folder_path
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

        else:
            print(f"\t\t Augment function {function_name} is not found.")
            img, bb = self.prepare_output(img=img, bb=bb)
            return img, bb

        img, bb = self.prepare_output(img=img, bb=bb)

        # for some function need to replace original image
        try:    
            if function_parameter["REPLACE"]:
                self.save_img(img=img, path=self.img_path)
                self.save_bb(bb_list=bb, path=self.bb_path)
        except:
            pass

        return img, bb
