from utils import Constants, Utils
from icecream import ic
import cv2
import numpy as np

class AugmentedGaugeModel:
    def __init__(
        self,
        match_img_bb_path=None,
        gauge_type=None,
        source_folder=None,
    ):
        self.match_img_bb_path = match_img_bb_path
        self.gauge_type = gauge_type
        self.source_folder = source_folder

        # TODO: count number of image augmented
        self.num_digital_aug = 0
        self.num_dial_aug = 0
        self.num_number_aug = 0
        self.num_level_aug = 0
        self.num_clock_aug = 0

    def augmented(
        self,
    ):
        if self.gauge_type is None or self.gauge_type == "":
            print(f"\t\t[X] PLEASE SPECIFIC GAUGE TYPE")
            return

        if self.match_img_bb_path == None:
            print(f"\t\t[X] PLEASE PASS IMAGE PATH AND LABELS PATH")
            return

        # TODO: augment 
        target_dict = {
            value: index
            for index, value in enumerate(
                Constants.map_data_dict[self.gauge_type]["target"]
            )
        }
        labels = list(target_dict.keys())

        for index, (img_path, bb_path) in enumerate(self.match_img_bb_path):
            
            if ((index+1) % 300 ==0):
                print(f"--- augmeent image: {index +1} ---")

            img = Utils.load_img_cv2(filepath=img_path)
            bb = Utils.load_bb(filepath=bb_path)

            target_size = [img.shape[0], img.shape[1]]

            if self.gauge_type == Constants.GaugeType.digital.value:
                self.num_digital_aug += self.augmented_digital(
                    gauge_name=Constants.GaugeType.digital.value,
                    start_index=self.num_digital_aug,
                    target_size=target_size,
                    img=img,
                    bb=bb,
                    labels=labels,
                    original_image_path=img_path,
                    original_label_path=bb_path,
                )
                # try:
                #     self.num_digital_aug += self.augmented_digital(
                #         gauge_name=Constants.GaugeType.digital.value,
                #         start_index=self.num_digital_aug,
                #         target_size=target_size,
                #         img=img,
                #         bb=bb,
                #         labels=labels,
                #         original_image_path=img_path,
                #         original_label_path=bb_path,
                #     )
                #      
                #     
                # except:
                #     print(f"\t--> can not augment {img_path}")

            if self.gauge_type == Constants.GaugeType.dial.value:
                self.augmented_dial(gauge_name=Constants.GaugeType.dial.value)

            if self.gauge_type == Constants.GaugeType.number.value:
                # self.num_number_aug += self.augmented_number(gauge_name=Constants.GaugeType.number.value)
                # TODO: same augment with digital
                self.num_number_aug += self.augmented_digital(
                    gauge_name=Constants.GaugeType.number.value,
                    start_index=self.num_number_aug,
                    target_size=target_size,
                    img=img,
                    bb=bb,
                    labels=labels,
                    original_image_path=img_path,
                    original_label_path=bb_path
                )
                

            if self.gauge_type == Constants.GaugeType.level.value:
                self.augmented_level(gauge_name=Constants.GaugeType.level.value)

            if self.gauge_type == Constants.GaugeType.clock.value:
                self.augmented_clock(gauge_name=Constants.GaugeType.clock.value)

    def change_original_gray(self, img,bb,transform, format=None):
        
        return

    def augmented_digital(
        self,
        gauge_name,
        start_index=0,
        target_size=None,
        img=None,
        bb=[],
        labels=None,
        original_image_path=None,
        original_label_path=None,
    ):
        from utils.constants import Constants

        # print(f"\t\t[-] AUGMENTED DIGITAL : {start_index}")
        original_transform = Utils.albu_grayscale(format=Constants.BoundingBoxFormat.YOLOV8)

        transform = Utils.albu_augmented_digital(
            target_size=target_size, format=Constants.BoundingBoxFormat.YOLOV8
        )

        aug_img_bb = Utils.get_output_from_transform(
            transform=transform, img=img, bb=bb, number_samples=2
        )
        
        for index, (aug_img, aug_bb) in enumerate(aug_img_bb):
            # Utils.visualize_img_bb(img=img, bb=bb, with_class=True, labels=labels)
            
            # TODO: save image
            new_img_name = original_image_path.with_suffix("").name + f"_aug_{index}"
            new_img_path = Utils.change_file_name(old_file_name=original_image_path, new_name=new_img_name, isrename=False)
            
            Utils.save_image(img=aug_img , filepath=new_img_path) # FIXME: uncomments this 
            
            # TODO: save bb
            new_label_name = original_label_path.with_suffix("").name + f"_aug_{index}"
            new_label_path = Utils.change_file_name(old_file_name=original_label_path, new_name=new_label_name, isrename=False)
            
            Utils.overwrite_label(txt_file_path=new_label_path, bb=aug_bb) # FIXME: uncomment this
            
            
            # if index < 10:
                # Utils.visualize_img_bb(img=Utils.load_img_cv2(filepath=new_img_path), bb=Utils.load_bb(filepath=new_label_path), with_class=True, labels=labels)
                # Utils.visualize_img_bb(img=ori_img_bb[0][0], bb=ori_img_bb[0][1], with_class=True, labels=labels)

        #TODO: augment original image for gray scale
        ori_img_bb =Utils.get_output_from_transform(
            transform=original_transform, img=Utils.load_img_cv2(filepath=original_image_path), bb=Utils.load_bb(filepath=original_label_path), number_samples=1
        )
        #TODO: delete original image and save new image
        Utils.delete_file(file_path=original_image_path) # delete folder
        Utils.save_image(img=ori_img_bb[0][0], filepath=original_image_path)
        #TODO: delete original bb and save new bb
        # Utils.delete_file(file_path=original_label_path)
        Utils.overwrite_label(txt_file_path=original_label_path,bb=ori_img_bb[0][1])
        # Utils.visualize_img_bb(img=ori_img_bb[0][0], bb=ori_img_bb[0][1], with_class=True, labels=labels)
            
        return start_index + len(aug_img_bb)

    def augmented_dial(self, gauge_name):
        print(f"\t\t[-] AUGMENTED DIAL")

    def augmented_number(self, gauge_name):
        print(f"\t\t[-] AUGMENTED NUMBER")
        from utils.constants import Constants
        
    def augmented_level(
        self,
    ):
        print(f"\t\t[-] AUGMENTED LEVEL")

    def augmented_clock(self, gauge_name):
        print(f"\t\t[-] AUGMENTED CLOCK")