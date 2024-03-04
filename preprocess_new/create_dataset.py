from icecream import ic
from pathlib import Path
from utils import Utils, Constants
import random


class CreateDataset:
    def __init__(self, data_yaml):
        from preprocess_new.combind import Combind
        from preprocess_new.preprocess import Preprocess
        from preprocess_new.augment import Augment

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
            preprocess_dict=self.preprocess_dict,
            dataset_path=Path(self.dataset_dict["FINAL_DATASET_PATH"]),
        )

        print(f"[-] AUGMENTED DATASET")
        self.augment = Augment(
            augment_dict=self.augment_dict,
            dataset_path=Path(self.dataset_dict["FINAL_DATASET_PATH"]),
        )
