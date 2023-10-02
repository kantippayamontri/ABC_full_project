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

    def conduct_dataset(
        self,
    ):
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
                # TODO: create dataset folder in datasets_for_train/digital ...
                Utils.delete_folder_mkdir(
                    PreprocessConstants.train_folder_dict[key], remove=True
                )
                # TODO: create train,val,test folder
                train_folder_path = (
                    PreprocessConstants.train_folder_dict[key] / Constants.train_folder
                )
                val_folder_path = (
                    PreprocessConstants.train_folder_dict[key] / Constants.val_folder
                )
                test_folder_path = (
                    PreprocessConstants.train_folder_dict[key] / Constants.test_folder
                )

                Utils.delete_folder_mkdir(train_folder_path, remove=False)
                Utils.delete_folder_mkdir(
                    train_folder_path / Constants.image_folder, remove=False
                )
                Utils.delete_folder_mkdir(
                    train_folder_path / Constants.label_folder, remove=False
                )

                Utils.delete_folder_mkdir(val_folder_path, remove=False)
                Utils.delete_folder_mkdir(
                    val_folder_path / Constants.image_folder, remove=False
                )
                Utils.delete_folder_mkdir(
                    val_folder_path / Constants.label_folder, remove=False
                )

                Utils.delete_folder_mkdir(test_folder_path, remove=False)
                Utils.delete_folder_mkdir(
                    test_folder_path / Constants.image_folder, remove=False
                )
                Utils.delete_folder_mkdir(
                    test_folder_path / Constants.label_folder, remove=False
                )

                # TODO: create data yaml file in datasets_for_train/
                data_yaml_dict = Utils.make_data_yaml_dict(
                    nc=len(Constants.map_data_dict[key]["target"]),
                    names=Constants.map_data_dict[key]["target"],
                )
                Utils.write_yaml(
                    data=data_yaml_dict,
                    filepath=PreprocessConstants.train_folder_dict[key]
                    / Constants.data_yaml_file,
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

            for folder in folders:  # TODO: Loop each folder in main dataset
                print(f"\t\t[-] CHECK AT {folder.name}")
                self.move_images_reclass_from_folder(
                    source_folder=folder, target_folder=target_path, key=key
                )

    def move_images_reclass_from_folder(self, source_folder, target_folder, key):
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

        # TODO: check folder to prepare to check class

        if data_yaml_path.exists():
            print(f"\t\t\t[/] FIND DATA YAML FILE")
            if self.check_yaml_is_ok(data_yaml_path=data_yaml_path):
                found_data_yaml = True
            else:
                print(f"\t\t\t\t[X] DATA YAML FILE HAVE PROBLEM")
        else:
            print(f"\t\t\t[X] CANNOT FIND DATA YAML FILE")

        if not found_data_yaml or (
            found_dict[Constants.train_folder] == False
            and found_dict[Constants.train_folder] == False
            and found_dict[Constants.train_folder] == False
        ):
            print(f"\t\t\t\t[X] CANNOT USE THIS FOLDER")
            return

        print(f"\t\t\t\t[-] CHECK FOR RECLASS")
        # TODO: check class
        # read data.yaml file
        data_yaml_file_source = Utils.read_yaml_file(data_yaml_path)
        data_yaml_file_target = Utils.read_yaml_file(
            PreprocessConstants.train_folder_dict[key] / Constants.data_yaml_file
        )

        if Utils.check_2_dataset_classe_index_ismatch(
            dataset_dict1=data_yaml_file_source,
            dataset_dict2=data_yaml_file_target,
        ):
            print(f"\t\t\t\t\t[-] NO NEED TO RECLASSES")

        else:
            print(f"\t\t\t\t[-] NEED TO RECLASSES")
            # TODO: loop in train, val and test folder
            for _train_val_test, _is_found in found_dict.items():
                if not _is_found:
                    continue

                # TODO: reclass images
                self.reclasses(
                    source_folder=source_folder / _train_val_test,
                    bb_before_dict=data_yaml_file_source,
                    bb_after_dict=data_yaml_file_target,
                )

        # reclasse and move images and labels

        # TODO: move images and labels file to datasets_for_train
        for _train_val_test, _is_found in found_dict.items():
            if not _is_found:
                continue

            # TODO: move images and labels
            self.move_images_and_labels(
                source_folder=source_folder / _train_val_test,
                target_folder=target_folder / Constants.train_folder,
            )  # ! FIXME: move to dataset_for_train/digital/train

    def reclasses(self, source_folder, bb_before_dict, bb_after_dict):
        # print(f"--- RECLASS FUNCTION ---")
        # print(f"source folder: {str(source_folder)}")
        # print(f"bb before")
        # print(bb_before_dict)
        # print(f"bb after")
        # print(bb_after_dict)

        bb_before = Utils.make_list_to_dict_index_value(bb_before_dict["names"])
        bb_after = Utils.make_list_to_dict_index_value(bb_after_dict["names"])

        print(bb_before)
        print(bb_after)

        filename_bb_list = Utils.get_filename_bb_folder(
            img_path=source_folder / Constants.image_folder,
            bb_path=source_folder / Constants.label_folder,
            source_folder=source_folder,
        )

        # TODO: reclass bb
        print(f"number of filename_bb_list: {len(filename_bb_list)}")

        for index, (img_path, bb_path) in enumerate(filename_bb_list):
            new_bb = Utils.reclass_bb_from_dict(
                bb=Utils.load_bb(bb_path),
                bb_dict_before=bb_before,
                bb_dict_after=bb_after,
            )
            Utils.overwrite_label(txt_file_path=bb_path, bb=new_bb)

        

        # for i in filename_bb_list[:30]:
        #     Utils.visualize_img_bb(img=Utils.load_img_cv2(i[0]), bb=Utils.load_bb(i[1]),with_class=True,labels=bb_after_dict["names"])

    def move_images_and_labels(self, source_folder, target_folder):
        print(f"source_folder: {source_folder}")
        print(f"target_folder: {target_folder}")
        pass

    def check_yaml_is_ok(self, data_yaml_path):
        yaml_file = Utils.read_yaml_file(str(data_yaml_path))
        # TODO: check key exists in dict
        if (
            (yaml_file.get("train") is not None and yaml_file.get("train") != "")
            and (yaml_file.get("val") is not None and yaml_file.get("val") != "")
            and (yaml_file.get("test") is not None and yaml_file.get("test") != "")
            and (yaml_file.get("nc") is not None and yaml_file.get("nc") != 0)
            and (yaml_file.get("names") is not None and yaml_file.get("names") != [])
        ):
            return True
        return False
