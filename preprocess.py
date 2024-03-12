# import preprocess
from utils import Constants, Utils
import argparse
from icecream import ic
from pathlib import Path
import preprocess

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="for training arguments parser")

# # Add positional argument
parser.add_argument("config", type=str, help="add config file in .yml")

# # Parse the command-line arguments
args = parser.parse_args()

# read config file
data_yaml = Utils.read_yaml_file(args.config)
dataset_dict = data_yaml["DATASET"]
preprocess_dict = data_yaml["PREPROCESS"]
augment_dict = data_yaml["AUGMENT"]

# check final dataset
if not Utils.check_folder_exists(Path(dataset_dict["FINAL_DATASET_PATH"])):
    ic(f"final folder no exist. -> make folder {dataset_dict['FINAL_DATASET_PATH']}")
    Utils.delete_folder_mkdir(
        folder_path=dataset_dict["FINAL_DATASET_PATH"], remove=False
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
        ic(f"dataset folder : {ds_folder} folder not exits.")
        exit()
    else:
        ic(f"dataset folder : {ds_folder} folder exits.")

# Preprocess
preprocess = preprocess.CreateDataset(data_yaml=data_yaml)