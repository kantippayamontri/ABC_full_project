from torch.utils.data import Dataset
from pathlib import Path
from utils import Constants, Utils, DatasetFromRoboflow
from train.train_constants import TrainConstants
import random
import yaml
import typing


class YOLODataset(Dataset):
    def __init__(self, dataset_type: Constants.GaugeType):
        super().__init__()
        print("\t[-] Initialized dataset")
        self.dataset_type = dataset_type
        self.train_img_bb = []
        self.val_img_bb = []
        self.test_img_bb = []
        self.target_labels = []
        self.data_yaml_path = None

        # TODO: chekc folder that ready to train
        print(f"\t\t[-] Check Dataset Folder")
        if self.check_folder():
            print(f"\t\t[/] Check Dataset Folder Successfully")
        else:
            print(f"\t\t[X] Check Dataset Folder Fail")

       

        print(f"---- preprocess part success ----")

    def check_folder(
        self,
    ):
        # TODO: check dataset_for_train folder
        if Utils.check_folder_exists(TrainConstants.train_dataset_root):
            # TODO: check gauge dataset ex dataset_for_train/digital
            if Utils.check_folder_exists(
                TrainConstants.train_dataset_path_dict[self.dataset_type]
            ):
                print(
                    f"\t\t\t[-] {str(TrainConstants.train_dataset_path_dict[self.dataset_type])} exists"
                )
                # TODO: check Train Val Test folder
                train_folder = (
                    TrainConstants.train_dataset_path_dict[self.dataset_type]
                    / Constants.train_folder
                )
                test_folder = (
                    TrainConstants.train_dataset_path_dict[self.dataset_type]
                    / Constants.test_folder
                )
                val_folder = (
                    TrainConstants.train_dataset_path_dict[self.dataset_type]
                    / Constants.val_folder
                )

                train_test_val_folder = [train_folder, val_folder, test_folder]
                for idx, folder in enumerate(train_test_val_folder):
                    if Utils.check_folder_exists(folder):
                        print(f"\t\t\t\t[-] {str(folder)} exists")
                        image_folder = folder / Constants.image_folder
                        label_folder = folder / Constants.label_folder

                        image_folder_exists = Utils.check_folder_exists(image_folder)
                        label_folder_exists = Utils.check_folder_exists(label_folder)

                        if image_folder_exists:
                            print(f"\t\t\t\t\t[/] Image folder exists")
                        else:
                            print(f"\t\t\t\t\t[X] Image folder not exists")
                            return False

                        if label_folder_exists:
                            print(f"\t\t\t\t\t[/] Label folder exists")
                        else:
                            print(f"\t\t\t\t\t[X] Label folder not exists")
                            return False

                        if image_folder_exists and label_folder_exists:
                            image_filenames = Utils.get_filenames_folder(image_folder)
                            label_filenames = Utils.get_filenames_folder(label_folder)

                            if (image_filenames != []) and (label_filenames != []):
                                img_bb_list = Utils.match_img_bb_filename(
                                    img_filenames_list=image_filenames,
                                    bb_filenames_list=label_filenames,
                                )

                                print(
                                    f"\t\t\t\t\t\t-> number of data: {len(img_bb_list)}"
                                )
                            else:
                                img_bb_list = []
                                print(f"\t\t\t\t\t\t-> number of data: {0}")

                            if idx == 0:  # TODO: train
                                self.train_img_bb = img_bb_list
                            elif idx == 1:  # TODO: val
                                self.val_img_bb = img_bb_list
                            elif idx == 2:  # TODO: test
                                self.test_img_bb = img_bb_list

                    else:
                        return False

                # TODO: check data.yaml file
                yaml_path = (
                    TrainConstants.train_dataset_path_dict[self.dataset_type]
                    / Constants.data_yaml_file
                )
                if Utils.check_yaml(yaml_path=yaml_path):
                    print(f"\t\t\t\t[/] YAML file is OK")
                    self.target_labels = Utils.read_yaml_file(yaml_file_path=yaml_path)["names"]
                    self.data_yaml_path = yaml_path
                else:
                    print(f"\t\t\t\t[X] YAML file not OK")
                    return False

                return True
            else:
                print(
                    f"\t\t\t---> {str(TrainConstants.train_dataset_path_dict[self.dataset_type])} not exists"
                )

        return False

    def __getitem__(
        self,
        index,
    ):
        return None

    def __len__(
        self,
    ):
        n_train = len(self.train_img_bb)
        n_val = len(self.val_img_bb)
        n_test = len(self.test_img_bb)
        
        print(f"number of train: {n_train}")
        print(f"number of val: {n_val}")
        print(f"number of test: {n_test}")
        
        return n_train + n_val + n_test

    def change_folder_name4train(self, old_folder_name, new_folder_name):
        return

    # def prepare_train(self):
    #     print(f"self.target_folder_path: {self.target_folder_path}")
    #     Utils.delete_folder_mkdir(
    #         self.target_folder_path, remove=True
    #     )  # create train_dataset folder
    #     # Utils.deleted_folder(self.target_folder_path)
    #     if Utils.check_folder_exists(self.dataset.dataset_folder):
    #         if Utils.check_folder_exists(self.target_folder_path_with_folder_name):
    #             Utils.deleted_folder(self.target_folder_path_with_folder_name)

    #         Utils.copy_folder(
    #             source_folder=self.dataset.dataset_folder,
    #             target_folder=self.target_folder_path_with_folder_name,
    #         )

    #         return Utils.change_folder_name(
    #             old_folder_name=Constants.train_dataset_folder / self.key,
    #             new_folder_name="datasets",
    #         )

    #     else:
    #         print(f"--- does not have base folder ---")
    #         return None

    # def get_data_yaml(
    #     self,
    # ):
    #     return self.data_yaml_path

    def get_data_yaml4train(
        self,
    ):
        return self.data_yaml_path

    # def get_all_files_bb(self, source_folder):
    #     if not Utils.check_folder_exists(source_folder):
    #         return []

    #     img_files_list = Utils.get_filename_bb_folder(
    #         source_folder=source_folder,
    #         img_path=source_folder / Constants.image_folder,
    #         bb_path=source_folder / Constants.label_folder,
    #     )
    #     return img_files_list

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
            if number_of_samples > len(self.train_img_bb):
                print(
                    f"can't show train -> number of samples({number_of_samples}) > train size({len(self.train_img_bb)})"
                )
            else:
                random_index = list(
                    [
                        random.randint(a=0, b=len(self.train_img_bb))
                        for _ in range(number_of_samples)
                    ]
                )
                data = list(
                    [self.train_img_bb[index] for index in random_index]
                )

        elif datasetTrainValTest == Constants.DatasetTrainValTest.VAL:
            print("show validation")
            if number_of_samples > len(self.val_img_bb):
                print(
                    f"can't show val -> number of samples({number_of_samples}) > val size({len(self.val_img_bb)})"
                )
            else:
                random_index = list(
                    [
                        random.randint(a=0, b=len(self.val_img_bb))
                        for _ in range(number_of_samples)
                    ]
                )
                data = list(
                    [self.val_img_bb[index] for index in random_index]
                )
        elif datasetTrainValTest == Constants.DatasetTrainValTest.TEST:
            print("show test")
            if number_of_samples > len(self.test_img_bb):
                print(
                    f"can't show test -> number of samples({number_of_samples}) > test size({len(self.test_img_bb)})"
                )
            else:
                random_index = list(
                    [
                        random.randint(a=0, b=len(self.test_img_bb))
                        for _ in range(number_of_samples)
                    ]
                )
                data = list(
                    [self.test_img_bb[index] for index in random_index]
                )

        # visualize the images
        for index, (img_path, bb_path) in enumerate(data):
            img = Utils.load_img_cv2(filepath=img_path)
            bb = Utils.load_bb(filepath=bb_path)
            Utils.visualize_img_bb(
                img=img, bb=bb, with_class=True, labels=self.target_labels
            )
            # print(f"target_labels: {self.target_labels}")

    def change_settings_dataset(
        self,
    ):
        setting_yaml_path = Constants.setting_yaml
        read_setting = Utils.read_yaml_file(setting_yaml_path)
        datasets_dir = (
            Constants.dataset_dir_setting
        )  # / "dataset" / "train_dataset" #/ "dataset"
        read_setting["datasets_dir"] = str(datasets_dir)  # + "/"
        Utils.write_yaml(data=read_setting, filepath=setting_yaml_path)
        print(f"--- write setting success ---")
