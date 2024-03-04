from icecream import ic
from utils import Utils, Constants


class Augment:
    def __init__(self, augment_dict, dataset_path):
        self.augment_dict = augment_dict
        self.dataset_path = dataset_path

        ic(self.augment_dict)
        ic(self.dataset_path)

        for folder in self.augment_dict["FOLDER"]:
            print(f"\t[-] AUGMENT AT {str(dataset_path / folder)}")
            self.augment(
                augment_list=self.augment_dict["AUGMENT_LIST"],
                dataset_folder=dataset_path / folder,
                number_augment=self.augment_dict["NUMBER_AUGMENT"],
            )

    def augment(self, augment_list=[], dataset_folder=None, number_augment=1):
        ic(augment_list, dataset_folder)

        if augment_list == []:
            print(f"\t\t[X] NO NEED TO AUGMENT -> AUGMENT LIST = 0")
            return

        if dataset_folder is None:
            print(f"\t\t[X] AUGMENT FAIL -> NO DATASET FOLDER")
            return

        matches_img_bb = Utils.get_filename_bb_folder(
                img_path=dataset_folder / "images", bb_path=dataset_folder / "labels"
            )
        
        for img_path, bb_path in matches_img_bb:
            img = Utils.load_img_cv2(filepath=img_path)
            bb = Utils.load_bb(filepath=bb_path)
            
            for round in range(number_augment):
                

        # (function_name, function_parameter) = tuple(
        #     (key, value) for key, value in pre_d.items()
        # )[0]

        # print(f"\t\t[-] {function_name}")

