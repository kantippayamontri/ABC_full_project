import preprocess
from utils import Constants, Utils
import argparse
from icecream import ic

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="for training arguments parser")

# Add positional argument
parser.add_argument("--input_file",type=str, help = "your python file to run")
parser.add_argument("--dataset_type", type=str, help="type of dataset to train")
parser.add_argument("--dataset_target", type=str, help="path of the target dataset folder", default="")

# Parse the command-line arguments
args = parser.parse_args()

#check dataset target folder is exists.
if not Utils.check_folder_exists(args.dataset_target) and args.dataset_target != "":
    Utils.delete_folder_mkdir(args.dataset_target, remove=True)

# Access the values of the arguments
input_file = args.input_file
dataset_type = Utils.get_enum_by_value(value=args.dataset_type.lower(),enum=Constants.GaugeType)
# 
datasetCombineModel = preprocess.DatasetCombineModel(make_frame=False, dataset_choose=dataset_type, target_dataset_folder=args.dataset_target, )
datasetCombineModel.conduct_dataset(delete_dataset_for_train=True, )
# datasetCombineModel.visualize_samples(gauge_type=dataset_type, number_of_samples=10)

ic(args.input_file)
ic(args.dataset_type)
ic(args.dataset_target)


