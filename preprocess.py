# import preprocess
from utils import Constants, Utils
import argparse
from icecream import ic
from pathlib import Path
import preprocess_new

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="for training arguments parser")

# # Add positional argument
parser.add_argument("config", type=str, help="add config file in .yml")

# # Parse the command-line arguments
args = parser.parse_args()

# ic(f"config path: {args.config}")

# read config file
data_yaml = Utils.read_yaml_file(args.config)
dataset_dict = data_yaml["DATASET"]
preprocess_dict = data_yaml["PREPROCESS"]
augment_dict = data_yaml["AUGMENT"]

# ic(data_yaml)
# ic(dataset_dict)
# ic(preprocess_dict)
# ic(augment_dict)

# ic(f"final path: {dataset_dict['FINAL_DATASET_PATH']}")

# check final dataset
if not Utils.check_folder_exists(Path(dataset_dict["FINAL_DATASET_PATH"])):
    ic(f"final folder no exist. -> make folder {dataset_dict['FINAL_DATASET_PATH']}")
    Utils.delete_folder_mkdir(
        folder_path=dataset_dict["FINAL_DATASET_PATH"], remove=False
    )
else:
    ic(
        f"final folder exist. -> remove folder {dataset_dict['FINAL_DATASET_PATH']} and recreate"
    )
    Utils.delete_folder_mkdir(
        folder_path=dataset_dict["FINAL_DATASET_PATH"], remove=True
    )
    #create train folder
    Utils.delete_folder_mkdir(str(Path(dataset_dict["FINAL_DATASET_PATH"]) / Constants.train_folder),remove=False)
    Utils.delete_folder_mkdir(str(Path(dataset_dict["FINAL_DATASET_PATH"]) / Constants.train_folder / Constants.image_folder),remove=False) # create images folder
    Utils.delete_folder_mkdir(str(Path(dataset_dict["FINAL_DATASET_PATH"]) / Constants.train_folder / Constants.label_folder ),remove=False) # create labels folder

    #create valid folder
    Utils.delete_folder_mkdir(str(Path(dataset_dict["FINAL_DATASET_PATH"]) / Constants.val_folder),remove=False)
    Utils.delete_folder_mkdir(str(Path(dataset_dict["FINAL_DATASET_PATH"]) / Constants.val_folder / Constants.image_folder),remove=False)
    Utils.delete_folder_mkdir(str(Path(dataset_dict["FINAL_DATASET_PATH"]) / Constants.val_folder / Constants.label_folder),remove=False)
    #create test folder
    Utils.delete_folder_mkdir(str(Path(dataset_dict["FINAL_DATASET_PATH"]) / Constants.test_folder),remove=False)
    Utils.delete_folder_mkdir(str(Path(dataset_dict["FINAL_DATASET_PATH"]) / Constants.test_folder / Constants.image_folder),remove=False)
    Utils.delete_folder_mkdir(str(Path(dataset_dict["FINAL_DATASET_PATH"]) / Constants.test_folder / Constants.label_folder),remove=False)

    #create yaml file
    data_yaml_dict = Utils.make_data_yaml_dict(
        nc=len(Constants.map_data_dict[dataset_dict["DATASET_TYPE"]]["target"]),
        names=Constants.map_data_dict[dataset_dict["DATASET_TYPE"]]["target"],
    )
    #write yaml file
    Utils.write_yaml(
        data=data_yaml_dict,
        filepath=Path(dataset_dict["FINAL_DATASET_PATH"]) / Constants.data_yaml_file,
    )

# check dataset folder exits
for ds_folder in dataset_dict["DATASET_PATH"]:
    if not Utils.check_folder_exists(Path(ds_folder)):
        ic(f"datset folder : {ds_folder} folder not exits.")
        exit()
    else:
        ic(f"datset folder : {ds_folder} folder exits.")

# Preprocess
preprocess = preprocess_new.CreateDataset(data_yaml=data_yaml)
# preprocess = preprocess_new.Preprocess(preprocess_dict=preprocess_dict,final_folder=dataset_dict["FINAL_DATASET_PATH"])



# # Add positional argument
# parser.add_argument("--input_file",type=str, help = "your python file to run")
# parser.add_argument("--dataset_type", type=str, help="type of dataset to train")
# parser.add_argument("--dataset_target", type=str, help="path of the target dataset folder", default="")

# # Parse the command-line arguments
# args = parser.parse_args()

# #check dataset target folder is exists.
# if not Utils.check_folder_exists(args.dataset_target) and args.dataset_target != "":
#     Utils.delete_folder_mkdir(args.dataset_target, remove=True)

# # Access the values of the arguments
# input_file = args.input_file
# dataset_type = Utils.get_enum_by_value(value=args.dataset_type.lower(),enum=Constants.GaugeType)
# #
# datasetCombineModel = preprocess.DatasetCombineModel(make_frame=False, dataset_choose=dataset_type, target_dataset_folder=args.dataset_target, )
# datasetCombineModel.conduct_dataset(delete_dataset_for_train=True, )
# # datasetCombineModel.visualize_samples(gauge_type=dataset_type, number_of_samples=10)

# ic(args.input_file)
# ic(args.dataset_type)
# ic(args.dataset_target)
