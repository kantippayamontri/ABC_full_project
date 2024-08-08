from icecream import ic
from utils import Utils
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
import os

class Preprocess:
    def __init__(self,dataset_type="digital", preprocess_dict=None, dataset_path=None, data_yaml_path=None):
        
        self.dataset_type = dataset_type
        self.preprocess_dict = preprocess_dict
        self.target_folder = dataset_path
        self.data_yaml_path = data_yaml_path

        self.number_of_error =0

        ic(self.data_yaml_path)
        ic(self.preprocess_dict,self.target_folder)


        for folder in self.preprocess_dict["FOLDER"]: 
            print(f'\t[-] PREPROCESS AT {str(dataset_path / folder)}')
            self.preprocess(
                preprocess_list=self.preprocess_dict["PREPROCESS_LIST"],
                dataset_folder=dataset_path / folder,
            )

    def preprocess(self, preprocess_list=[], dataset_folder=None, remove_original=False):
        from .transforms import Transform

        if preprocess_list == []:
            print(f"\t\t[X] NO NEED TO PREPROCESS -> PREPROCESS LIST = 0")
            return

        if dataset_folder is None:
            print(f"\t\t[X] PREPROCESS FAIL -> NO DATASET FOLDER")
            return

        count_original_file = Utils.count_files( folder=dataset_folder / "images")
        if count_original_file ==0:
            print(f"No image in this folder")
            return
        original_filename_bb = ((img_path, bb_path) for (img_path, bb_path) in Utils.get_filename_bb_folder( img_path=dataset_folder / "images", bb_path=dataset_folder / "labels"))
        # print(original_filename_bb)
        

        for pre_d in preprocess_list:

            # matches_img_bb = Utils.get_filename_bb_folder( img_path=dataset_folder / "images", bb_path=dataset_folder / "labels")
            (function_name, function_parameter) = tuple(
                (key, value) for key, value in pre_d.items()
            )[0]

            number_files = Utils.count_files(folder=dataset_folder / "images")

            if number_files ==0:
                print(f"\t\t[X] PREPROCESS FAIL -> NUMBER OF FILE 0")
                return

            matches_img_bb_gen = ((img_path, bb_path) for (img_path, bb_path) in Utils.get_filename_bb_folder( img_path=dataset_folder / "images", bb_path=dataset_folder / "labels"))
            
            # print(f"size of matches list : {sys.getsizeof(matches_img_bb, 'bytes')}")
            # print(f"size of matches gen : {sys.getsizeof(matches_img_bb_gen, 'bytes')}")
            # print(f"number of files: {number_files}")


            print(f"\t\t[-] {function_name}")

            
            
            for (img_path, bb_path) in tqdm(matches_img_bb_gen, total=number_files):
                self.process_image(img_path, bb_path, function_name, function_parameter, dataset_folder, dataset_type=self.dataset_type)
            
            if function_name == "CROP" and ("REMOVE_ORIGINAL" in function_parameter.keys()):
                if function_parameter["REMOVE_ORIGINAL"]:
                    for (img_path, bb_path) in tqdm(original_filename_bb, total=count_original_file):
                        if Path(img_path).is_file():
                            os.remove(str(Path(img_path)))
                        
                        if Path(bb_path).is_file():
                            os.remove(str(Path(bb_path)))
            
            if function_name == "CLOCK" and "ADD_NEEDLE" in function_parameter.keys():
                if function_parameter["ADD_NEEDLE"]:
                    data_yaml_dict = Utils.read_yaml_file(yaml_file_path=self.data_yaml_path)
                    # add needle to names
                    names = data_yaml_dict["names"]
                    names.append("needle") # add class needle
                    data_yaml_dict["names"] = names
                    # add needle to class
                    nc = int(data_yaml_dict["nc"])
                    nc += 1 # add clas needle
                    data_yaml_dict["nc"] = nc
                    Utils.write_yaml(data=data_yaml_dict, filepath=self.data_yaml_path) # write the yaml file
                
                
           
    
    def process_image(self,img_path, bb_path, function_name, function_parameter, dataset_folder, dataset_type=None):
        from .transforms import Transform
        img = Utils.load_img_cv2(filepath=img_path)
        bb = Utils.load_bb(filepath=bb_path)

        if (img is None) or (bb is None):
            return

        transform = Transform(img_path=img_path, bb_path=bb_path)
        img, bb = transform.transform_dict_function(
            function_name=function_name,
            function_parameter=function_parameter,
            img=img.copy(),
            bb=bb.copy(),
            target_folder_path=dataset_folder,
            dataset_type=dataset_type
        )

        
