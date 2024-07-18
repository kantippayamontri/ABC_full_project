from icecream import ic
from pathlib import Path
from utils import Utils, Constants
import random


class CreateDataset:
    def __init__(self, data_yaml):
        from preprocess.combind import Combind
        from preprocess.preprocess import Preprocess
        from preprocess.augment import Augment

        self.data_yaml = data_yaml
        self.dataset_dict = data_yaml["DATASET"]
        self.preprocess_dict = data_yaml["PREPROCESS"]
        self.augment_dict = data_yaml["AUGMENT"]

        ic(self.data_yaml)

        # combind the dataset to final dataset path
        print("[-] COMBIND DATASET")
        self.combind = Combind(data_dict=self.dataset_dict)
        self.combind.combind()
        # self.combind.visualize_samples(n=3, folder="valid")

        # preprocess the dataset in final dataset_path
        # preprocess -> all image in finale dataset_path folder
        print("[-] PREPROCESS DATASET")
        self.preprocess = Preprocess(
            dataset_type=self.dataset_dict["DATASET_TYPE"],
            preprocess_dict=self.preprocess_dict,
            dataset_path=Path(self.dataset_dict["FINAL_DATASET_PATH"]),
            data_yaml_path=Path(self.dataset_dict["FINAL_DATASET_PATH"] ) / "data.yaml"
        )

        print(f"[-] AUGMENTED DATASET")
        self.augment = Augment(
            dataset_type=self.dataset_dict["DATASET_TYPE"],
            augment_dict=self.augment_dict,
            dataset_path=Path(self.dataset_dict["FINAL_DATASET_PATH"]),
        )

        print(f"[-] NUMBER OF FILE AFTER")
        for folder in ["train", "valid", "test"]:
            folder_path = Path(self.dataset_dict["FINAL_DATASET_PATH"]) / folder /"images"
            number_of_file = 0
            if Utils.check_folder_exists(folder_path=folder_path):
                number_of_file = Utils.count_files(folder=folder_path)
            print(f"\t[-] Number of {folder} : {number_of_file} images")