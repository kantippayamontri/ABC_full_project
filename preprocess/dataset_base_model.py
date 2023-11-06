from utils import Utils
from utils import Constants
from preprocess.preprocess_constants import PreprocessConstants
import random
from icecream import ic
from pathlib import Path


class DatasetCombineModel:
    def __init__(
        self,
        make_frame=False,
    ):
        self.found_folder_dict = {
            Constants.GaugeType.digital.value: False,
            Constants.GaugeType.dial.value: False,
            Constants.GaugeType.number.value: False,
            Constants.GaugeType.clock.value: False,
            Constants.GaugeType.level.value: False,
        }

        self.make_frame = make_frame

    def conduct_dataset(self, delete_dataset_for_train=True):
        # self.check_folder(delete_dataset_for_train=delete_dataset_for_train)
        # print()
        # print("-" * 100)
        # print()
        # self.import_datasets_roboflow()  # TODO: download dataset from roboflow
        # print()
        # print("-" * 100)
        # print()
        # self.combine_datasets()
        # print()
        # print("-" * 100)
        # print()
        if self.make_frame:
            self.make_frame_dataset()
            print()
            print("-" * 100)
            print()
        self.divide_datasets(
            train_ratio=PreprocessConstants.train_ratio
        )  # TODO: divide into train set and validation set
        print()
        print("-" * 100)
        print()
        self.preprocess()
        print()
        print("-" * 100)
        print()
        self.augmented()

    def check_folder(self, delete_dataset_for_train):
        for key, value in PreprocessConstants.base_folder_dict.items():
            print(f"[/] CHECK {key} GAUGE")
            if Utils.check_folder_exists(value) or (
                key == "digital"
            ):  # FIXME: remove key==
                print(f"\t[/] FOLDER FOUND at {value}")
                # TODO: create dataset folder in datasets_for_train/digital ...
                Utils.delete_folder_mkdir(
                    PreprocessConstants.train_folder_dict[key],
                    remove=delete_dataset_for_train,
                )
                # print(
                #     f"delete at {PreprocessConstants.train_folder_dict[key]}, remove: {delete_dataset_for_train}"
                # )
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

    def import_datasets_roboflow(
        self,
    ):
        print(f"--- IMPORT DATASETS FROM ROBOFLOW ---")
        return

    def make_frame_dataset(
        self,
    ):
        print(f"--- in function make_frame_dataset ---")
        # TODO: define dataset that use for crop images
        make_frame_dataset_path = [
            PreprocessConstants.train_folder_dict[Constants.GaugeType.digital.value]
        ]

        # TODO: check make_frame dataset
        dataset_frame_path = Path("./dataset_frame")
        Utils.delete_folder_mkdir(folder_path=dataset_frame_path, remove=True)

        for path in make_frame_dataset_path:
            yaml_path = path / Constants.data_yaml_file
            data_yaml = Utils.read_yaml_file(yaml_file_path=yaml_path)
            # ic(data_yaml)

            if "frame" not in data_yaml["names"]:
                continue

            frame_index = data_yaml["names"].index("frame")

            root_path = path / Constants.train_folder
            images_path = root_path / Constants.image_folder
            labels_path = root_path / Constants.label_folder

            img_bb_match = Utils.get_filename_bb_folder(
                img_path=images_path,
                bb_path=labels_path,
            )

            for idx, (img_path, bb_path) in enumerate(img_bb_match):
                img = Utils.load_img_cv2(filepath=img_path)
                bb = Utils.load_bb(filepath=bb_path)

                if idx % 300 ==0:
                    print(f"---> make frame : {idx+1}/{len(img_bb_match)}")
                
                frame_img = Utils.crop_one_class(
                    img=img,
                    bb=bb,
                    need_resize=True,
                    target_size=[img.shape[0], img.shape[1]],
                    add_pixels=random.randint(0, 10),
                    class_crop=frame_index,
                )  # TODO: return onlu frame img
                

                for index, img in enumerate(frame_img):
                    new_name = Utils.generate_random_string(25) + f"_{index}" + ".jpg"
                    # TODO: save images
                    Utils.save_image(img=img, filepath=dataset_frame_path / new_name)
                
                # if idx == 10:
                #     return

        return

    def divide_datasets(
        self,
        train_ratio=0.8,
    ):
        # print(f"train ratio: {train_ratio}")
        # print(f"val ratio: {1 - train_ratio}")

        for key, value in PreprocessConstants.base_folder_dict.items():
            # if not self.found_folder_dict[key]: # FIXME: uncomment this statements
            #     continue
            # else:
            #     print(f"[-] DIVIDE {key} DATASET")

            if key != Constants.GaugeType.digital.value:
                continue

            target_source_folder = PreprocessConstants.train_folder_dict[key]
            target_train_folder = (
                PreprocessConstants.train_folder_dict[key] / Constants.train_folder
            )
            target_val_folder = (
                PreprocessConstants.train_folder_dict[key] / Constants.val_folder
            )

            target_val_image_folder = target_val_folder / Constants.image_folder
            target_val_label_folder = target_val_folder / Constants.label_folder

            # print(target_source_folder)
            # print(target_train_folder)
            # print(target_val_folder)

            # TODO: get all images and labels file
            train_img_bb = Utils.get_filename_bb_folder(
                img_path=target_train_folder / Constants.image_folder,
                bb_path=target_train_folder / Constants.label_folder,
                source_folder=None,
            )

            random.shuffle(train_img_bb)  # TODO: shuffle the list
            index_divide_val = int((len(train_img_bb) - 1) * (1 - train_ratio))

            val_img_bb = train_img_bb[:index_divide_val]

            # TODO: move image and labe from train to val
            for idx, (img_path, lb_path) in enumerate(val_img_bb):
                # print(f"img path: {str(img_path)}, lb_path: {str(lb_path)}")

                # TODO: move images
                Utils.move_file(
                    source_file_path=img_path,
                    target_file_path=target_val_image_folder / img_path.name,
                )

                # TODO: move labels
                Utils.move_file(
                    source_file_path=lb_path,
                    target_file_path=target_val_label_folder / lb_path.name,
                )

        return

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

            for index, folder in enumerate(
                folders
            ):  # TODO: Loop each folder in main dataset
                print(f"\t\t[-] CHECK AT {folder.name}")
                self.move_images_reclass_from_folder(
                    source_folder=folder,
                    target_folder=target_path,
                    key=key,
                    index=index,
                )

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
            # print(str(source_folder / k))
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

        # print(f"data_yaml_file_source: {data_yaml_file_source}")
        # print(f"data_yaml_file_target: {data_yaml_file_target}")

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

        # TODO: move images and labels file to datasets_for_train
        for _train_val_test, _is_found in found_dict.items():
            if not _is_found:
                continue

            # TODO: move images and labels
            self.move_images_and_labels(
                source_folder=source_folder / _train_val_test,
                target_folder=target_folder / Constants.train_folder,
                gauge=key,  # gauge use for rename,
                folder_number=index,  # folder number for rename
            )  # ! FIXME: move to dataset_for_train/digital/train

        #     Utils.deleted_folder(source_folder.parent)  # deleted folder after # FIXME: use this

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

        # TODO: reclass bb
        # print(f"number of filename_bb_list: {len(filename_bb_list)}")

        for index, (img_path, bb_path) in enumerate(filename_bb_list):
            new_bb = Utils.reclass_bb_from_dict(
                bb=Utils.load_bb(bb_path),
                bb_dict_before=bb_before,
                bb_dict_after=bb_after,
            )
            Utils.overwrite_label(txt_file_path=bb_path, bb=new_bb)
            Utils.write_yaml(data=target_yaml_data, filepath=target_yaml_path)

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

            # print(f"new_source_image_path: {str(new_source_image_path)}")

            target_image = (
                target_folder / Constants.image_folder / new_source_image_path.name
            )

            # Utils.move_folder(
            #     source_folder=new_source_image_path, target_folder=target_image
            # )
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
            # Utils.move_folder(
            #     source_folder=new_source_label_path, target_folder=target_label
            # )
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

    def visualize_samples(self, gauge_type, number_of_samples=1):
        from preprocess.preprocess_constants import PreprocessConstants

        # TODO: get filenames and bb and labels
        source_folder = (
            PreprocessConstants.train_folder_dict[Constants.GaugeType.digital.value]
            / Constants.train_folder
        )
        img_path = source_folder / Constants.image_folder
        bb_path = source_folder / Constants.label_folder
        match_filename_bb = Utils.get_filename_bb_folder(
            img_path=img_path, bb_path=bb_path, source_folder=source_folder
        )
        # print(f"--- Match File ---")
        # TODO: random images and bb
        number_of_images = len(match_filename_bb)
        random_index_list = []
        for i in range(number_of_samples):
            index = random.randint(0, number_of_images)
            random_index_list.append(index)

        # TODO: visulize image and bb
        labels = Constants.map_data_dict[Constants.GaugeType.digital.value]["target"]
        for index in random_index_list:
            _img_path = match_filename_bb[index][0]
            _bb_path = match_filename_bb[index][1]
            Utils.visualize_img_bb(
                img=Utils.load_img_cv2(filepath=_img_path),
                bb=Utils.load_bb(filepath=_bb_path),
                with_class=True,
                labels=labels,
            )
        return

    def preprocess(
        self,
    ):  # use for crop images
        from preprocess.preprocess_constants import PreprocessConstants
        from preprocess.preprocess_model import PreprocessGaugeModel

        print(f"[-] Preprocess Dataset")
        for key, value in PreprocessConstants.train_folder_dict.items():
            # print(f"key: {key}, value: {value}")

            source_folder = (
                PreprocessConstants.train_dataset_folder / key / Constants.train_folder
            )
            source_image_folder = source_folder / Constants.image_folder
            source_label_folder = source_folder / Constants.label_folder

            if not Utils.check_folder_exists(source_folder):
                continue
            else:
                print(f"\t[-] preprocess at {key}")

            match_images_labels = Utils.get_filename_bb_folder(
                img_path=source_image_folder,
                bb_path=source_label_folder,
                source_folder=source_folder,
            )

            # TODO: preprocess data
            preprocessmodel = PreprocessGaugeModel(
                match_img_bb_path=match_images_labels,
                gauge_type=key,
                source_folder=source_folder,
            )
            preprocessmodel.preprocess()

        return

    def augmented(
        self,
    ):
        from preprocess.preprocess_constants import PreprocessConstants
        from preprocess.augment_model import AugmentedGaugeModel

        print(f"[-] augmented Dataset")
        for key, value in PreprocessConstants.train_folder_dict.items():
            # print(f"key: {key}, value: {value}")

            source_folder = (
                PreprocessConstants.train_dataset_folder / key / Constants.train_folder
            )
            source_image_folder = source_folder / Constants.image_folder
            source_label_folder = source_folder / Constants.label_folder

            if not Utils.check_folder_exists(source_folder):
                continue
            else:
                print(f"\t[-] augmented at {key}")

            match_images_labels = Utils.get_filename_bb_folder(
                img_path=source_image_folder,
                bb_path=source_label_folder,
                source_folder=source_folder,
            )

            # TODO: augmented data

            augmentedmodel = AugmentedGaugeModel(
                match_img_bb_path=match_images_labels,
                gauge_type=key,
                source_folder=source_folder,
            )
            augmentedmodel.augmented()

            # TODO: preprocess data
            # preprocessmodel = ProprocessGaugeModel(match_img_bb_path=match_images_labels,gauge_type=key,source_folder=source_folder)
            # preprocessmodel.preprocess()
