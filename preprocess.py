import preprocess
from utils import Constants, Utils
import argparse
from icecream import ic

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="for training arguments parser")

# Add positional argument
parser.add_argument("input_file",type=str, help = "your python file to run")
parser.add_argument("dataset_type", type=str, help="type of dataset to train")

# Parse the command-line arguments
args = parser.parse_args()
# Access the values of the arguments
input_file = args.input_file
dataset_type = Utils.get_enum_by_value(value=args.dataset_type.lower(),enum=Constants.GaugeType)
# 
datasetCombineModel = preprocess.DatasetCombineModel(make_frame=False, dataset_choose=dataset_type)
datasetCombineModel.conduct_dataset(delete_dataset_for_train=True, )
# datasetCombineModel.visualize_samples(gauge_type=dataset_type, number_of_samples=10)

