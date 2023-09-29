from utils import Utils
from utils import Constants
from preprocess.preprocess_constants import PreprocessConstants


class DatasetBaseModel:
    def __init__(
        self,
    ):
        self.found_folder_dict = {
            Constants.GaugeType.digital.value: False,
            Constants.GaugeType.dial.value: False,
            Constants.GaugeType.number.value: False,
            Constants.GaugeType.clock.value: False,
            Constants.GaugeType.level.value: False,
        }
        
    
    def conduct_dataset(self,):
        self.check_folder()
        print()
        print("-" * 50)
        print()
        self.combine_datasets()
        print()
        print("-" * 50)
        print()

    def check_folder(self):
        for key, value in PreprocessConstants.base_folder_dict.items():
            print(f"[/] CHECK {key} GAUGE")
            if Utils.check_folder_exists(value):
                print(f"\t[/] FOLDER FOUND at {value}")
                Utils.delete_folder_mkdir(
                    PreprocessConstants.train_folder_dict[key], remove=True
                )
                self.found_folder_dict[key] = True

            else:
                print(f"\t[/] FOLDER NOT FOUND at {value}")
                self.found_folder_dict[key] = False

    def combine_datasets(
        self,
    ):
        for key, value in PreprocessConstants.base_folder_dict.items():
            if not self.found_folder_dict[key]:
                continue
            else:
                print(f"[-] COMBINE {key} DATASET")

            dataset_path = value
            target_path = PreprocessConstants.train_folder_dict[key]

            if dataset_path == None or target_path == None:
                print(f"\t[X] COMBINE DATASET UNSUCCESSFULL")

            print(f"\t[-] COMBINE DATASET at {dataset_path}")

            if not any(dataset_path.iterdir()):
                print(f"\t[X] COMBINE DATASET UNSUCCESSFULL -> FOLDER IS EMPTY")

            folders = [item for item in dataset_path.iterdir() if item.is_dir()]

            for folder in folders: # TODO: Loop each folder in main dataset
                print(f"\t\t[-] CHECK AT {folder.name}")
                self.move_images_reclass_from_folder(
                    source_folder=folder, target_folder=target_path
                )

    def move_images_reclass_from_folder(self, source_folder, target_folder):
        # print(f"move images from {source_folder} to {target_folder}")

        check_folder_dict = {
            Constants.train_folder: {
                Constants.image_folder: source_folder
                / Constants.train_folder
                / Constants.image_folder,
                Constants.label_folder: source_folder
                / Constants.train_folder
                / Constants.label_folder,
            },
            Constants.val_folder: {
                Constants.image_folder: source_folder
                / Constants.val_folder
                / Constants.image_folder,
                Constants.label_folder: source_folder
                / Constants.val_folder
                / Constants.label_folder,
            },
            Constants.test_folder: {
                Constants.image_folder: source_folder
                / Constants.test_folder
                / Constants.image_folder,
                Constants.label_folder: source_folder
                / Constants.test_folder
                / Constants.label_folder,
            },
        }

        data_yaml_path = source_folder / Constants.data_yaml_file

        found_dict = {
            Constants.train_folder: True,
            Constants.val_folder: True,
            Constants.test_folder: True,
        }
        
        found_data_yaml = False

        # TODO: check train , valid and test folder has images and labels

        for k, _ in check_folder_dict.items():
            if (
                not Utils.check_folder_exists(str(source_folder / k))
                or not any(check_folder_dict[k][Constants.image_folder].iterdir())
                or not any(check_folder_dict[k][Constants.label_folder].iterdir())
            ):
                found_dict[k] = False
                print(f"\t\t\t[X] CAN NOT FIND {k}")
            else:
                print(f"\t\t\t[/] FIND {k}")

        # TODO: check for reclass 
        
        if data_yaml_path.exists():
            print(f"\t\t\t[/] FIND DATA YAML FILE")
            found_data_yaml = True
        else:
            print(f"\t\t\t[X] CANNOT FIND DATA YAML FILE")
            
        if not found_data_yaml or (found_dict[Constants.train_folder]==False and found_dict[Constants.train_folder]==False and found_dict[Constants.train_folder]==False):
            print(f"\t\t\t\t[X] CANNOT USE THIS FOLDER")
            return
        else:
            pass
        
        
