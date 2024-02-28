from pathlib import Path
from utils import Utils, Constants
from icecream import ic
import random


class Combind:
    def __init__(self, data_dict):
        self.dataset_dict = data_dict

    def combind(
        self,
    ):
        data_dict = self.dataset_dict
        ic(data_dict)

        print(f"[-] COMBINE {data_dict['DATASET_TYPE']} DATASET")

        # move all images and bb to final_dataset_path/train
        for folder_num, ds_path in enumerate(data_dict["DATASET_PATH"]):

            dataset_path = Path(ds_path)
            target_path = Path(data_dict["FINAL_DATASET_PATH"])

            print(f"\t[-] COMBINE DATASET at {dataset_path}")

            if not any(dataset_path.iterdir()):
                print(f"\t[X] COMBINE DATASET UNSUCCESSFULL -> FOLDER IS EMPTY")
                continue

            self.move_images_reclass_from_folder(
                source_folder=dataset_path,
                target_folder=target_path,
                key=data_dict["DATASET_TYPE"],
                index=folder_num,
            )

        # move image from train to valid, test
        train_percent = data_dict["PERCENT_TRAIN"]
        val_percent = data_dict["PERCENT_VAL"]
        test_percent = data_dict["PERCENT_TEST"]

        if train_percent + val_percent + test_percent != 100:
            print(f"[X] train, valid, test not equal to 100")

        filename_bb = Utils.get_filename_bb_folder(
            img_path=Path(data_dict["FINAL_DATASET_PATH"]) / "train" / "images",
            bb_path=Path(data_dict["FINAL_DATASET_PATH"]) / "train" / "labels",
        )
        random.shuffle(filename_bb)  # random order of image and bb
        number_file = len(filename_bb)
        train_index = int((train_percent / 100.0) * number_file)
        val_index = train_index + int((val_percent / 100.0) * number_file)

        train_filename_bb = filename_bb[:train_index]
        val_filename_bb = filename_bb[train_index:val_index]
        test_filename_bb = filename_bb[val_index:]

        print(f"\t[-] number of train images : " + str(len(train_filename_bb)))
        print(f"\t[-] number of valid images : " + str(len(val_filename_bb)))
        print(f"\t[-] number of test images : " + str(len(test_filename_bb))) 

        for folder_move in ["valid", "test"]:
            if folder_move == "valid":
                fn = val_filename_bb
            elif folder_move == "test":
                fn = test_filename_bb

            for _, (img_p, bb_p) in enumerate(fn):
                Utils.move_file(
                    source_file_path=img_p,
                    target_file_path=Path(data_dict["FINAL_DATASET_PATH"])
                    / folder_move
                    / "images",
                )  # move image
                Utils.move_file(
                    source_file_path=bb_p,
                    target_file_path=Path(data_dict["FINAL_DATASET_PATH"])
                    / folder_move
                    / "labels",
                )  # move bb

    def move_images_reclass_from_folder(
        self, source_folder, target_folder, key, index=0
    ):  # index use for rename images -> make images unique
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
            target_folder / Constants.data_yaml_file
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
                    target_yaml_path=data_yaml_path,
                    target_yaml_data=data_yaml_file_target,
                )
            print(f"\t\t\t\t\t[/] RECLASSES SUCCESSFUL")

        # TODO: move images and labels file to final folder
        for _train_val_test, _is_found in found_dict.items():
            if not _is_found:
                continue

            # TODO: move images and labels
            self.move_images_and_labels(
                source_folder=source_folder / _train_val_test,
                target_folder=target_folder / "train",
                gauge=key,  # gauge use for rename,
                folder_number=index,  # folder number for rename
            )

    def move_images_and_labels(
        self, source_folder, target_folder, gauge=None, folder_number=None
    ):
        match_img_bb = Utils.get_filename_bb_folder(
            img_path=source_folder / Constants.image_folder,
            bb_path=source_folder / Constants.label_folder,
            source_folder=None,
        )

        for index, (_img, _bb) in enumerate(match_img_bb):
            # TODO: rename image
            new_name = Utils.new_name_with_date(
                gauge=gauge, number=index + 1, folder_number=folder_number
            )

            # TODO: move images
            new_source_image_path = Utils.change_file_name(
                old_file_name=_img, new_name=new_name
            )

            target_image = (
                target_folder / Constants.image_folder / new_source_image_path.name
            )

            Utils.copy_file(
                source_file_path=new_source_image_path, target_file_path=target_image
            )  # TODO: use copy file instead of move

            # TODO: move labels
            new_source_label_path = Utils.change_file_name(
                old_file_name=_bb, new_name=new_source_image_path.stem
            )
            target_label = (
                target_folder / Constants.label_folder / new_source_label_path.name
            )
            Utils.copy_file(
                source_file_path=new_source_label_path, target_file_path=target_label
            )  # TODO: use copy file instead of move

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

    def reclasses(
        self,
        source_folder,
        bb_before_dict,
        bb_after_dict,
        target_yaml_path,
        target_yaml_data,
    ):
        bb_before = Utils.make_list_to_dict_index_value(bb_before_dict["names"])
        bb_after = Utils.make_list_to_dict_index_value(bb_after_dict["names"])

        filename_bb_list = Utils.get_filename_bb_folder(
            img_path=source_folder / Constants.image_folder,
            bb_path=source_folder / Constants.label_folder,
            source_folder=source_folder,
        )

        for index, (img_path, bb_path) in enumerate(filename_bb_list):
            new_bb = Utils.reclass_bb_from_dict(
                bb=Utils.load_bb(bb_path),
                bb_dict_before=bb_before,
                bb_dict_after=bb_after,
            )
            Utils.overwrite_label(txt_file_path=bb_path, bb=new_bb)

        Utils.write_yaml(data=target_yaml_data, filepath=target_yaml_path)

    def visualize_samples(self, n=5,folder="train"):
        Utils.visualize_samples(
            source_folder=Path(self.dataset_dict["FINAL_DATASET_PATH"]) / folder,
            number_of_samples=n,
            gauge_type=self.dataset_dict["DATASET_TYPE"],
        )
