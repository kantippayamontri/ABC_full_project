from icecream import ic
from typing import 

class Preprocess:
    def __init__(self, preprocess_dict, final_folder):
        self.pre_folder = preprocess_dict["FOLDER"]
        self.prefix = preprocess_dict["PREFIX"]
        self.pre_list = preprocess_dict["PREPROCESS_LIST"]
        self.final_folder = final_folder
    
    def check_folder(self,) :
        ...
        
    
    def preprocess(self,):
        ...