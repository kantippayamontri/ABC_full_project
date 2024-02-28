from icecream import ic
import typing

class Preprocess:
    def __init__(self, preprocess_dict, dataset_path):
        self.preprocess_dict = preprocess_dict
        self.target_folder = dataset_path

        ic(self.preprocess_dict)
        ic(self.target_folder)
        
        # for folder in self.preprocess_dict["FOLDER"]: #FIXME: choose this line instead
        for folder in ["valid"]:
            self.preprocess(preprocess_list=self.preprocess_dict["PREPROCESS_LIST"], dataset_folder=dataset_path / folder)
    
    def preprocess(self,preprocess_list=[], dataset_folder=None):
        #FIXME: uncomment this for 
        # if preprocess_list == []:
        #     print(f"\t[X] NO NEED TO PREPROCESS -> PREPROCESS LIST = 0")
        #     return

        # if dataset_folder is None:
        #     print(f"\t[X] PREPROCESS FAIL -> NO DATASET FOLDER")
        #     return
        
        print(preprocess_list)
        