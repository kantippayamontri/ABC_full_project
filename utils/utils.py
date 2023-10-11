import os
import shutil
import yaml
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import albumentations as A
from pathlib import Path
from PIL import Image
import datetime


class Utils:
    @staticmethod
    def check_folder_exists(folder_path):
        if os.path.exists(folder_path):
            return True
        return False

    @staticmethod
    def change_folder_name(old_folder_name, new_folder_name):
        # print(f"old_folder_name: {old_folder_name}")
        # print(f"new_folder_name: {new_folder_name}")
        # print(f"old_folder_name parent: {old_folder_name.parent}")

        new_folder_name_path = old_folder_name.parent / new_folder_name
        # print(f"new_folder_name_path: {new_folder_name_path}")
        os.rename(old_folder_name, new_folder_name_path)

        return new_folder_name_path

    @staticmethod
    def change_file_name(old_file_name, new_name, isrename=True):
        extension = old_file_name.suffix

        new_name_path = old_file_name.with_name(new_name).with_suffix(extension)

        if isrename:
            os.rename(old_file_name, new_name_path)

        return new_name_path

    @staticmethod
    def new_name_with_date(gauge=None, number=None, folder_number=None):
        current_datetime = datetime.datetime.now()
        current_date = current_datetime.date().strftime("%Y_%m_%d")
        new_name = current_date
        if gauge is not None:
            new_name = new_name + f"_{gauge}"
        if folder_number is not None:
            new_name = new_name + f"_{folder_number}"
        if number is not None:
            new_name = new_name + f"_{number}"
        return new_name

    @staticmethod
    def new_name_crop_with_date(
        gauge=None, number=None, class_crop=None, image_number=None
    ):
        from utils.utils import Utils

        main_name = Utils.new_name_with_date(
            gauge=gauge, number=number, folder_number=class_crop
        )
        if image_number is not None:
            main_name += f"_{image_number}"
        return main_name

    @staticmethod
    def deleted_folder(folder_path):
        shutil.rmtree(folder_path)

    @staticmethod
    def move_folder(source_folder, target_folder):
        shutil.move(str(source_folder), str(target_folder))
        
    @staticmethod
    def move_file(source_file_path, target_file_path):
        # print(f"source path: {source_file_path} \ntarget path: {target_file_path}")
        shutil.move(str(source_file_path), str(target_file_path))
        return

    @staticmethod
    def copy_folder(source_folder, target_folder):
        shutil.copytree(str(source_folder), str(target_folder))

    @staticmethod
    def get_filenames_folder(
        source_folder,
    ):
        # print(f"source folder: {str(source_folder)}")
        return [
            source_folder / file.name
            for file in source_folder.iterdir()
            if file.is_file()
        ]

    @staticmethod
    def get_filename_bb_folder(img_path=None, bb_path=None, source_folder=None):
        img_filenames = Utils.get_filenames_folder(img_path)
        bb_filenames = Utils.get_filenames_folder(bb_path)
        match_files = Utils.match_img_bb_filename(
            img_filenames_list=img_filenames,
            bb_filenames_list=bb_filenames,
            source_folder=source_folder,
        )
        return match_files

    @staticmethod
    def delete_folder_mkdir(folder_path, remove=False):
        if Utils.check_folder_exists(folder_path):
            if remove:
                shutil.rmtree(folder_path)
            else:
                print(f"--- This folder exists ---")
                return False
        os.makedirs(folder_path)
        return True

    @staticmethod
    def make_dataset_dict(
        version=None,
        api_key=None,
        model_format=None,
        project_name=None,
        dataset_folder=None,
        workspace=None,
        user_name=None,
    ):
        return {
            "version": version,
            "api_key": api_key,
            "model_format": model_format,
            "project_name": project_name,
            "dataset_folder": dataset_folder,
            "workspace": workspace,
            "user_name": user_name,
        }

    @staticmethod
    def read_yaml_file(yaml_file_path):
        # Open the YAML file for reading
        with open(yaml_file_path, "r") as yaml_file:
            # Parse the YAML data
            yaml_data = yaml.safe_load(yaml_file)
        return yaml_data
    
    @staticmethod
    def check_yaml(yaml_path=None,yaml_file=None): # TODO: you can pass yaml file or yaml path
        if (yaml_path is None and yaml_file is None) or (yaml_path is not None and yaml_file is not None):
            print(f"Please pass yaml file or yaml path or one of these argument")
        
        if yaml_path is not None:
            yaml_file = Utils.read_yaml_file(yaml_file_path=yaml_path)
        
        check_list = ['names', 'nc', 'test', 'train', 'val']
        
        for check in check_list:
            if check not in yaml_file:
                return False
        return True

    @staticmethod
    def write_yaml(data, filepath):
        with open(str(filepath), "w") as file:
            yaml.dump(data, file)

    @staticmethod
    def make_data_yaml_dict(nc, names):
        data_yaml_dict = {
            "train": "../train/images",
            "val": "../valid/images",
            "test": "../test/images",
            "nc": nc,
            "names": names,
        }
        return data_yaml_dict

    @staticmethod
    def check_2_dataset_classe_index_ismatch(dataset_dict1, dataset_dict2):
        dict_check = {"nc": None, "names": None}

        dict1_check = dict_check.copy()
        dict1_check["nc"] = dataset_dict1["nc"]
        dict1_check["names"] = dataset_dict1["names"]

        dict2_check = dict_check.copy()
        dict2_check["nc"] = dataset_dict2["nc"]
        dict2_check["names"] = dataset_dict2["names"]

        if dict1_check != dict2_check:
            return False

        return True

    @staticmethod
    def make_list_to_dict_index_value(data: list):
        return {value: index for index, value in enumerate(data)}

    @staticmethod
    def albu_crop_img_bb(img=None, bb_crop=None, format=None, add_pixels=0):
        from utils.constants import Constants
        max_x = img.shape[1] - 1
        max_y = img.shape[0] - 1 
        if (format == None) or (format == Constants.BoundingBoxFormat.YOLOV8):
            xyxy_crop = Utils.change_format_yolo2xyxy(
                img_size=img.shape, bb=bb_crop, with_class=True
            )["bb"]
            transform = A.Compose(
                [
                    A.Crop(
                        x_min=max(0, xyxy_crop[0][0] - add_pixels),
                        y_min=max(0, xyxy_crop[0][1] - add_pixels),
                        x_max=min(max_x, xyxy_crop[1][0] + add_pixels),
                        y_max=min(max_y, xyxy_crop[1][1] + add_pixels),
                    ),
                ],
                bbox_params={
                    "format": "yolo",
                },
            )
        return transform

    @staticmethod
    def albu_resize_img_bb(target_size=None, format=None):
        from utils.constants import Constants

        target_width = target_size[0]
        target_height = target_size[1]

        if (format == None) or (format == Constants.BoundingBoxFormat.YOLOV8):
            transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=max(target_size)),
                    A.PadIfNeeded(
                        min_height=target_height,
                        min_width=target_width,
                        border_mode=cv2.BORDER_CONSTANT,
                    ),
                    A.Resize(
                        height=target_height, width=target_width, always_apply=True
                    ),
                ],
                bbox_params={
                    "format": "yolo",
                },
            )

        return transform

    @staticmethod
    def get_output_from_transform(transform=None, img=None, bb=None, number_samples=1):
        if (transform is None) or (img is None) or (bb is None):
            print(f"*** CANNOT AUGMENTED IMAGE SOME PARAMETER IS NONE***")
            return None

        aug_images_bb_list = []

        for i in range(number_samples):
            # TODO: reformat from yolo to albumentation for transformation
            permutation = [1, 2, 3, 4, 0]
            bb_albu = bb[:, permutation]

            # TODO: transform image
            img_bb_trans = transform(image=img, bboxes=bb_albu)  # transform images

            # TODO: get image and bb after transform
            img_trans = img_bb_trans["image"]  # image after transformation
            bb_trans = img_bb_trans["bboxes"]  # bb after transformation
            bb_trans = np.array([list(_bb) for _bb in bb_trans])

            # TODO: reformat from albumentation to yolo
            if len(bb_trans) != 0:
                inversePermutation = [4, 0, 1, 2, 3]
                bb_trans = bb_trans[:, inversePermutation]

                aug_images_bb_list.append((img_trans, bb_trans))

            else:
                aug_images_bb_list.append(img_trans, np.array([]))

        return aug_images_bb_list

    # TODO: albu for augment digital
    @staticmethod
    def albu_augmented_digital(target_size, format=None):
        from utils.constants import Constants

        target_width = target_size[0]
        target_height = target_size[1]

        if (format == None) or (format == Constants.BoundingBoxFormat.YOLOV8):
            transform = A.Compose(
                [
                    # TODO: augmentation images
                    A.ChannelShuffle(
                        p=0.7,
                    ),  # ! channel shuffle
                    A.MultiplicativeNoise(
                        multiplier=[0.4, 1.0], elementwise=True, p=0.5
                    ),
                    A.Blur(blur_limit=(7,7), p=0.5),
                    # A.GaussNoise(var_limit=(0,0.075), mean=0, p=1.0), # FIXME: not work
                    A.Flip(p=0.7,), # ! flip verical or horizontal
                    A.Rotate(limit=[-15,15], border_mode=cv2.BORDER_CONSTANT,p=0.7),
                    A.ColorJitter(p=0.7),
                    
                    # TODO: resize and padding images if needed
                    A.LongestMaxSize(max_size=max(target_size)),
                    A.PadIfNeeded(
                        min_height=target_height,
                        min_width=target_width,
                        border_mode=cv2.BORDER_CONSTANT,
                    ),
                    A.Resize(height=target_height, width=target_width),
                ],
                bbox_params={
                    "format": "yolo",
                },
            )

        return transform

    @staticmethod
    def calculate_overlap_percentage(box1, box2):
        x1_1, y1_1 = box1[0]
        x2_1, y2_1 = box1[1]
        x1_2, y1_2 = box2[0]
        x2_2, y2_2 = box2[1]

        # Calculate the coordinates of the intersection
        intersection_x1 = max(x1_1, x1_2)
        intersection_y1 = max(y1_1, y1_2)
        intersection_x2 = min(x2_1, x2_2)
        intersection_y2 = min(y2_1, y2_2)

        # Calculate the area of intersection
        intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(
            0, intersection_y2 - intersection_y1 + 1
        )

        # Calculate the areas of both bounding boxes
        area_box1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
        area_box2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

        # Calculate the percentage of overlap
        overlap_percentage = (
            intersection_area / float(area_box1 + area_box2 - intersection_area)
        ) * 100

        return overlap_percentage

    @staticmethod
    def calculate_intersection(box1, box2):
        x1_1, y1_1 = box1[0]
        x2_1, y2_1 = box1[1]
        x1_2, y1_2 = box2[0]
        x2_2, y2_2 = box2[1]

        # Calculate the coordinates of the intersection
        intersection_x1 = max(x1_1, x1_2)
        intersection_y1 = max(y1_1, y1_2)
        intersection_x2 = min(x2_1, x2_2)
        intersection_y2 = min(y2_1, y2_2)

        # Check if there's an actual intersection
        if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
            # return (intersection_x1, intersection_y1, intersection_x2, intersection_y2)
            return True
        else:
            return False  # No intersection

    @staticmethod
    def crop_img(
        img=None, bb=None, class_crop=None, need_resize=True, target_size=None,add_pixels=0,class_ignore=None
    ):
        from utils.constants import Constants

        full_img = img.copy()
        bb_temp = bb.copy()

        bb_crop = []
        bb = []

        for index, _bb in enumerate(bb_temp):
            if int(_bb[0]) == int(class_crop):
                bb_crop.append(bb_temp[index])
                
            bb.append(bb_temp[index])

        bb = np.array(bb)

        # crop the images
        crop_images_bb_list = []
        for index_bb_crop, _bb_crop in enumerate(bb_crop):
            bb_use = []
            img = full_img.copy() 
                
            for _bb in bb:
                box1 = Utils.change_format_yolo2xyxy(
                    img_size=img.shape, bb=_bb_crop, with_class=True
                )["bb"]
                box2 = Utils.change_format_yolo2xyxy(
                    img_size=img.shape, bb=_bb, with_class=True
                )["bb"]
                # print(f"check intersect: " + str(Utils.calculate_intersection(box1, box2)))
                if Utils.calculate_intersection(box1, box2):
                    bb_use.append(_bb.copy())

            bb_use = np.array(bb_use)

            if len(bb_use) != 0:
                # if len(bb_crop) > 1:
                # Utils.visualize_img_bb(img=img, bb=bb, with_class=True)
                # print(f"bb_use: {bb_use}")
                transform_crop_image = Utils.albu_crop_img_bb(
                    img=img, bb_crop=_bb_crop, format=Constants.BoundingBoxFormat.YOLOV8, add_pixels=add_pixels
                )

                # format for albumentation yolo = [cx,cy,w,h,class]
                permutation = [1, 2, 3, 4, 0]
                bb_use = bb_use[:, permutation]

                transformed = transform_crop_image(image=img, bboxes=bb_use)
                img = transformed["image"]
                bb_use = transformed["bboxes"]
                bb_use = np.array([list(bb_temp) for bb_temp in bb_use])

                if need_resize:
                    if target_size == None:
                        print(f"--- cannot resize image ---")
                    else:
                        transform_resize_image = Utils.albu_resize_img_bb(
                            target_size=target_size,
                        )
                        transformed = transform_resize_image(image=img, bboxes=bb_use)
                        img = transformed["image"]
                        bb_use = transformed["bboxes"]
                        bb_use = np.array([list(bb_temp) for bb_temp in bb_use])

                # from format [cx,cy,w,h,class] to [class,cx,cy,w,h]
                if len(bb_use) != 0:
                    inversePermutation = [4, 0, 1, 2, 3]
                    bb_use = bb_use[:, inversePermutation]
                    
                if class_ignore is not None:
                    bb_use = np.array([bb for bb in bb_use if int(bb[0]) != class_ignore])

                crop_images_bb_list.append((img, bb_use))
            else:
                crop_images_bb_list.append((img, []))

            # if len(crop_images_bb_list) > 1:
            #     print(f"bb_Crop: {bb_crop}")
            #     print(f"bb: {bb}")
            #     for d in crop_images_bb_list:
            #         Utils.visualize_img_bb(img=d[0], bb=d[1], with_class=True)

            return crop_images_bb_list

        return crop_images_bb_list

    @staticmethod
    def resize_img_bb(target_size=[1280, 1280], img=None, bb=None):
        permutation = [1, 2, 3, 4, 0]
        bb = bb[:, permutation]

        transform_resize_image = Utils.albu_resize_img_bb(
            target_size=target_size,
        )
        transformed = transform_resize_image(image=img, bboxes=bb)
        img = transformed["image"]
        bb = transformed["bboxes"]
        bb = np.array([list(bb_temp) for bb_temp in bb])

        inversePermutation = [4, 0, 1, 2, 3]
        bb = bb[:, inversePermutation]

        return [img, bb]

    @staticmethod
    def make_crop_image_and_bb(img=None, bb=None, class_crop=None, **kwargs):
        new_img_bb = Utils.crop_img(
            img=img,
            bb=bb,
            class_crop=class_crop,
            target_size=kwargs["crop_target_size"],
        )
        # # need to reclass -> class_crop remove
        # for index, (img, bb) in enumerate(new_img_bb):
        #     reclass_bb = Utils.reclass_bb_from_crop(bb=bb, class_crop=class_crop)
        #     new_img_bb[index] = (img,reclass_bb)

        return new_img_bb

    # @staticmethod
    # def reclass_bb_from_crop(bb=None, class_crop=None):
    #     for index, _bb in enumerate(bb):
    #         if int(_bb[0]) > class_crop:
    #             bb[index][0] = int(bb[index][0]) - 1
    #     return bb

    @staticmethod
    def reclass_bb_from_dict(
        bb=None, bb_dict_before=None, bb_dict_after=None, class_crop=None
    ):
        value_to_key_before = {value: key for key, value in bb_dict_before.items()}

        # print(f"bb_dict_before: {bb_dict_before}")
        # print(f"bb_dict_after: {bb_dict_after}")
        # print(f"value_to_key_before: {value_to_key_before}")
        # print(f"class_crop: {class_crop}")
        bb_temp = []
        for index, _bb in enumerate(bb):
            key = value_to_key_before[int(_bb[0])]
            if key in list(bb_dict_after.keys()):
                new_value = bb_dict_after[key]
                # minus_for_crop =0
                if (class_crop != None) and (new_value > class_crop):
                    # print(f'use -> class_crop: {class_crop}, new_value: {new_value}')
                    # minus_for_crop -=1
                    pass
                bb[index][0] = int(new_value)  # + minus_for_crop
                bb_temp.append(bb[index])

        return bb_temp

    @staticmethod
    def change_format_yolo2xyxy(img_size=None, bb=None, with_class=False):
        # print(img_size, bb)
        img_w = img_size[1]
        img_h = img_size[0]
        nx = int(float(bb[1] * img_w))
        ny = int(float(bb[2] * img_h))
        nw = int(float(bb[3] * img_w))
        nh = int(float(bb[4] * img_h))
        if with_class:
            return {
                "class": bb[0],
                "bb": [
                    (nx - int(nw / 2), ny - int(nh / 2)),
                    (nx + int(nw / 2), ny + int(nh / 2)),
                ],
            }
        else:
            return {
                "class": None,
                "bb": [
                    (nx - int(nw / 2), ny - int(nh / 2)),
                    (nx + int(nw / 2), ny + int(nh / 2)),
                ],
            }

    @staticmethod
    def visualize_img_bb(img, bb, with_class=False, format=None, labels=None):
        from utils.constants import Constants

        xyxy_bb = []
        if (len(bb) != 0) and (
            format == Constants.BoundingBoxFormat.YOLOV8 or format == None
        ):
            xyxy_bb = [
                Utils.change_format_yolo2xyxy(
                    img_size=img.shape, bb=_bb, with_class=with_class
                )
                for _bb in bb
            ]

        plt.imshow(img)
        plt.axis("off")  # Turn off axes numbers and ticks

        for xyxy in xyxy_bb:
            color_index = 0
            top_left = (0, 0)
            bottom_right = (0, 0)
            if with_class:
                color_index = int(xyxy["class"])
                top_left = xyxy["bb"][0]
                bottom_right = xyxy["bb"][1]
            else:
                top_left = xyxy["bb"][0]
                bottom_right = xyxy["bb"][1]

            # create bounding box
            bbox = patches.Rectangle(
                xy=top_left,
                width=bottom_right[0] - top_left[0],
                height=bottom_right[1] - top_left[1],
                linewidth=2,
                edgecolor=np.array(Constants.colors[color_index]) / 255.0,
                facecolor="none",
            )

            # add the bounding box rectangle to the current plot
            plt.gca().add_patch(bbox)
            # add text to the bounding box
            if labels != None:
                plt.text(
                    top_left[0],
                    top_left[1] - 10,
                    labels[color_index],
                    color=np.array(Constants.colors[color_index]) / 255.0,
                )

        plt.show()

    @staticmethod
    def change_filename_sample(
        filepath, filename, index, start_index=0, extension=None
    ):
        # print(f"filename: {filename}")
        if index == start_index:
            if extension == None:
                return filepath
            else:
                return filepath.with_suffix(extension)
        else:
            if extension == None:
                ext = Path(filename).suffix
                filename_with_extension = Path(filename).stem + f"_{index}" + ext
                return filepath.parent / filename_with_extension
            else:
                filename_with_extension = Path(filename).stem + f"_{index}" + extension
                return filepath.parent / filename_with_extension

    @staticmethod
    def match_img_bb_filename(
        img_filenames_list=None, bb_filenames_list=None, source_folder=None
    ):
        from utils.constants import Constants

        match_img_bb = []
        bb_folder_path = ""

        if source_folder != None:
            bb_folder_path = source_folder / Constants.label_folder
        else:
            bb_folder_path = bb_filenames_list[0].parent

        for index, img_filename in enumerate(img_filenames_list):
            # find match bounding box
            filename = Path(img_filename)
            filename = Utils.change_filename_sample(
                filepath=filename, filename=None, index=0, extension=".txt"
            ).name
            label_full_path = bb_folder_path / filename

            if label_full_path.is_file():
                match_img_bb.append((img_filename, label_full_path))

        return match_img_bb

    @staticmethod
    def save_image(img, filepath):
        # Convert the NumPy array to a PIL Image
        pil_image = Image.fromarray(img)

        if filepath.is_file():
            os.remove(filepath)
        # Save the image
        pil_image.save(filepath)

    @staticmethod
    def load_img_cv2(filepath):
        img = cv2.imread(str(filepath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def load_bb(filepath):
        bb = []

        fp = open(filepath, "r")  # read the bounding box
        for c, line in enumerate(fp):
            bb.append(list(float(n) for n in line.split(" ")))

        bb = np.array(bb)
        return bb

    @staticmethod
    def overwrite_label(txt_file_path, bb):
        file = open(txt_file_path, "w")
        # Write new content to the file
        str_save = ""
        for _bb in bb:
            str_save += f"{int(_bb[0])} {_bb[1]} {_bb[2]} {_bb[3]} {_bb[4]}\n"
        file.write(str_save)
        file.close()

    @staticmethod
    def make_dict_roboflow_dataset(roboflow_dict):
        # from utils.constants import Constants
        print(f"--- pass ---\n")
        return roboflow_dict
        # return  {
        #     "dataset dict": Utils.make_dataset_dict(
        #         version=27,
        #         api_key=api_key,
        #         model_format=model_format,
        #         project_name=project_name,
        #         dataset_folder=dataset_folder / "type3" / "gauge_display_frame",
        #     ),
        #     "type": DatasetType.TYPE_3.value,
        #     "key": DatasetUse.TYPE_3_GAUGE_DISPLAY_FRAME.value,
        #     "parameters":{
        #         "image_size": [1280, 1280],
        #     }
        # },
