from icecream import ic
from utils import Utils, Constants
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed


class Augment:
    def __init__(self, augment_dict, dataset_path, dataset_type):
        self.augment_dict = augment_dict
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type

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
        from .transforms import Transform

        ic(augment_list, dataset_folder)

        if augment_list == []:
            print(f"\t\t[X] NO NEED TO AUGMENT -> AUGMENT LIST = 0")
            return

        if dataset_folder is None:
            print(f"\t\t[X] AUGMENT FAIL -> NO DATASET FOLDER")
            return

        # matches_img_bb = Utils.get_filename_bb_folder( img_path=dataset_folder / "images", bb_path=dataset_folder / "labels")
        number_files = Utils.count_files(folder=dataset_folder / "images")

        if number_files==0:
            print(f"\t\t[X] AUGMENT FAIL -> NUMBER OF FILE 0")
            return
        
        matches_img_bb_gen = ((img_path, bb_path) for (img_path, bb_path) in Utils.get_filename_bb_folder( img_path=dataset_folder / "images", bb_path=dataset_folder / "labels"))

        num_cores = -1 #Use all available CPU cores
        with Parallel(n_jobs=num_cores) as parallel:
            parallel( delayed(self.augment_image_parallel)(img_path, bb_path, number_augment, augment_list, dataset_folder) for (img_path, bb_path) in tqdm(matches_img_bb_gen, total=number_files)) 

        # for (img_path, bb_path) in tqdm(matches_img_bb_gen, total=number_files):

    def augment_image_parallel(self,img_path, bb_path, number_augment, augment_list, dataset_folder):
        from .transforms import Transform
        img = Utils.load_img_cv2(filepath=img_path)
        bb = Utils.load_bb(filepath=bb_path)

        if (img is None) or (bb is None):
            return

        for round in range(number_augment):
            new_img = img.copy()
            new_bb = bb.copy()

            try:
                for aug_d in augment_list:
                    (function_name, function_parameter) = tuple(
                        (key, value) for key, value in aug_d.items()
                    )[0]
                    # ic(function_name, function_parameter)
                    transform = Transform(img_path=img_path, bb_path=bb_path)
                    new_img, new_bb = transform.transform_dict_function(
                        function_name=function_name,
                        function_parameter=function_parameter,
                        img=new_img,
                        bb=new_bb,
                        target_folder_path=dataset_folder,
                        dataset_type=self.dataset_type
                    )
            except Exception as e:  
                print(f"Error: {e}" )
                return 

            # save augment image with index of number samples prefix
            new_name_img = transform.make_new_name(
                name=Path(transform.get_img_path()),
                function_name="aug",
                prefix=f"{round}",
            )
            new_name_bb = transform.make_new_name(
                name=Path(transform.get_bb_path()),
                function_name="aug",
                prefix=f"{round}",
            )

            transform.save_img(img=new_img, path=Path(new_name_img))
            transform.save_bb(bb_list=new_bb, path=Path(new_name_bb))

            # Utils.visualize_img_bb(
            #     img=new_img,
            #     bb=new_bb,
            #     with_class=True,
            #     format=None,
            #     labels=["gauge", "display", "frame"],
            # )

        # (function_name, function_parameter) = tuple(
        #     (key, value) for key, value in pre_d.items()
        # )[0]

        # print(f"\t\t[-] {function_name}")
