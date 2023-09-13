from torch.utils.data import Dataset
from pathlib import Path
from utils import Constants, Utils, DatasetFromRoboflow
import random
import yaml


class YOLODataset(Dataset):
    def __init__(self, dataset_dict, remove_exists=False):
        super().__init__()
        self.remove_exists = remove_exists
        self.dataset_dict = dataset_dict["dataset dict"]
        self.key = dataset_dict["key"]
        self.type = dataset_dict["type"]
        self.target_folder_path = Constants.train_dataset_folder
        self.target_folder_path_with_folder_name = (
            Constants.train_dataset_folder / self.key
            # Constants.train_dataset_folder / "dataset"
        )
        # self.target_folder_path_with_folder_name = Constants.train_dataset_folder / "dataset"
        self.dataset = DatasetFromRoboflow(
            version=self.dataset_dict["version"],
            api_key=self.dataset_dict["api_key"],
            project_name=self.dataset_dict["project_name"],
            model_format=self.dataset_dict["model_format"],
            dataset_folder=self.dataset_dict["dataset_folder"],
            type=self.type,
            key=self.key,
            remove_exist=self.remove_exists,
        )
        self.dataset.import_datasets()
        self.target_folder_path_with_folder_name = self.prepare_train()
        self.data_yaml_path = self.target_folder_path_with_folder_name / "data.yaml"
        
        self.train_img_bb_filenames = self.get_all_files_bb(
            source_folder=self.target_folder_path_with_folder_name
            / Constants.train_folder
        )
        self.val_img_bb_filenames = self.get_all_files_bb(
            source_folder=self.target_folder_path_with_folder_name
            / Constants.val_folder
        )

        self.test_img_bb_filenames = self.get_all_files_bb(
            source_folder=self.target_folder_path_with_folder_name
            / Constants.test_folder
        )
        
        print(f"self.data_yaml_path: {self.data_yaml_path}")
        self.target_labels = list(
            self.dataset.map_class_yaml(
                yaml_path=self.data_yaml_path,
                map_dict=Constants.map_data_dict[self.type][self.key],
            )["target"].keys()
        )
        
        # self.prepare_train()
        # self.data_yaml_path = self.target_folder_path_with_folder_name / "data.yaml"
        
        # self.data_yaml_path_train = Utils.change_folder_name()

        print(f"---- preprocess part success ----")

    def __getitem__(
        self,
        index,
    ):
        return None

    def __len__(self, index, include_train=True, include_val=True, include_test=True):
        data = {}
        if include_train:
            data["train"] = len(self.train_img_bb_filenames)

        if include_val:
            data["val"] = len(self.val_img_bb_filenames)

        if include_test:
            data["test"] = len(self.test_img_bb_filenames)

        return data
    
    def change_folder_name4train(self, old_folder_name, new_folder_name):
        return

    def prepare_train(self):
        print(f"self.target_folder_path: {self.target_folder_path}")
        Utils.delete_folder_mkdir(
            self.target_folder_path , remove=True
        )  # create train_dataset folder
        # Utils.deleted_folder(self.target_folder_path)
        if Utils.check_folder_exists(self.dataset.dataset_folder):
            if Utils.check_folder_exists(self.target_folder_path_with_folder_name):
                Utils.deleted_folder(self.target_folder_path_with_folder_name)

            Utils.copy_folder(
                source_folder=self.dataset.dataset_folder,
                target_folder=self.target_folder_path_with_folder_name,
            )
            
            return Utils.change_folder_name(old_folder_name=Constants.train_dataset_folder / self.key, new_folder_name="datasets")
            
        else:
            print(f"--- does not have base folder ---")
            return None

    # def get_data_yaml(
    #     self,
    # ):
    #     return self.data_yaml_path
    
    def get_data_yaml4train(self,):
        return self.data_yaml_path

    def get_all_files_bb(self, source_folder):
        if not Utils.check_folder_exists(source_folder):
            return []

        img_files_list = Utils.get_filename_bb_folder(
            source_folder=source_folder,
            img_path=source_folder / Constants.image_folder,
            bb_path=source_folder / Constants.label_folder,
        )
        return img_files_list

    def show_samples(
        self,
        number_of_samples=5,
        random_seed=42,
        datasetTrainValTest=Constants.DatasetTrainValTest.TRAIN,
    ):
        data = []
        random_index = []

        if datasetTrainValTest == Constants.DatasetTrainValTest.TRAIN:
            print("show training")
            if number_of_samples > len(self.train_img_bb_filenames):
                print(
                    f"can't show train -> number of samples({number_of_samples}) > train size({len(self.train_img_bb_filenames)})"
                )
            else:
                random_index = list(
                    [
                        random.randint(a=0, b=len(self.train_img_bb_filenames))
                        for _ in range(number_of_samples)
                    ]
                )
                data = list(
                    [self.train_img_bb_filenames[index] for index in random_index]
                )

        elif datasetTrainValTest == Constants.DatasetTrainValTest.VAL:
            print("show validation")
            if number_of_samples > len(self.val_img_bb_filenames):
                print(
                    f"can't show val -> number of samples({number_of_samples}) > val size({len(self.val_img_bb_filenames)})"
                )
            else:
                random_index = list(
                    [
                        random.randint(a=0, b=len(self.val_img_bb_filenames))
                        for _ in range(number_of_samples)
                    ]
                )
                data = list(
                    [self.val_img_bb_filenames[index] for index in random_index]
                )
        elif datasetTrainValTest == Constants.DatasetTrainValTest.TEST:
            print("show test")
            if number_of_samples > len(self.test_img_bb_filenames):
                print(
                    f"can't show test -> number of samples({number_of_samples}) > test size({len(self.test_img_bb_filenames)})"
                )
            else:
                random_index = list(
                    [
                        random.randint(a=0, b=len(self.test_img_bb_filenames))
                        for _ in range(number_of_samples)
                    ]
                )
                data = list(
                    [self.test_img_bb_filenames[index] for index in random_index]
                )

        # visualize the images
        for index, (img_path, bb_path) in enumerate(data):
            img = Utils.load_img_cv2(filepath=img_path)
            bb = Utils.load_bb(filepath=bb_path)
            Utils.visualize_img_bb(
                img=img, bb=bb, with_class=True, labels=self.target_labels
            )
            # print(f"target_labels: {self.target_labels}")
            

    def change_settings_dataset(self,):
        setting_yaml_path = Constants.setting_yaml
        read_setting = Utils.read_yaml_file(setting_yaml_path)
        datasets_dir = Constants.dataset_dir_setting #/ "dataset" / "train_dataset" #/ "dataset"
        read_setting['datasets_dir'] = str(datasets_dir) #+ "/"
        Utils.write_yaml(data=read_setting, filepath=setting_yaml_path)
        print(f"--- write setting success ---")

