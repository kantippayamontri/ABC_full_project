from icecream import ic
import typing
from utils import Utils

class Preprocess:
    def __init__(self, preprocess_dict, dataset_path):
        self.preprocess_dict = preprocess_dict
        self.target_folder = dataset_path

        # ic(self.preprocess_dict)
        # ic(self.target_folder)
        
        # for folder in self.preprocess_dict["FOLDER"]: #FIXME: choose this line instead
        for folder in ["valid"]:
            self.preprocess(preprocess_list=self.preprocess_dict["PREPROCESS_LIST"], dataset_folder=dataset_path / folder)
    
    def preprocess(self,preprocess_list=[], dataset_folder=None):
        from .transforms import Transform

        #FIXME: uncomment this for 
        if preprocess_list == []:
            print(f"\t[X] NO NEED TO PREPROCESS -> PREPROCESS LIST = 0")
            return

        if dataset_folder is None:
            print(f"\t[X] PREPROCESS FAIL -> NO DATASET FOLDER")
            return
        
        print(dataset_folder)
        print(preprocess_list)

        # get matches images and bb
        matches_img_bb = Utils.get_filename_bb_folder(img_path= dataset_folder / "images", bb_path= dataset_folder / "labels")
        print(f"number of matches : {len(matches_img_bb)}")

        albumentation_list = []
        for _pre in preprocess_list:
            print(_pre)
        
        for _, (img_path, bb_path) in enumerate(matches_img_bb) : 
            #FIXME: delete this for testing
            if _ < 5:
                print(img_path, bb_path)
            else:
                break
                
            img = Utils.load_img_cv2(filepath=img_path)
            bb = Utils.load_bb(filepath=bb_path)

            transform = Transform()
            for pre_d in preprocess_list:
                (function_name, function_parameter) = tuple((key,value) for key,value in pre_d.items())[0]
                img,bb = transform.transform_dict_function(function_name=function_name, function_parameter=function_parameter, img=img, bb=bb)
                


