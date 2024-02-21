from utils import Constants, Utils
from train import YOLODataset, YOLOModel, TrainParameters
import argparse
from icecream import ic
from pathlib import Path

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="for training arguments parser")

# Add positional argument
parser.add_argument("input_file", type=str, help="your python file to run")
parser.add_argument("train_yml", type=str, help="path to train.yml file")

# Parse the command-line arguments
args = parser.parse_args()

data_yaml = Utils.read_yaml_file(yaml_file_path=args.train_yml)
dataset_dict = data_yaml["DATASET"]
model_dict = data_yaml["MODEL"]
train_parameters_dict = data_yaml["TRAIN_PARAMETERS"]

def experiment_name(dataset_type=None, model_type=None, project_path=None):
    name = ""
    if dataset_type == None:
        name += "(ds_t)_None"
    else:
        name += f"(ds_t)_{dataset_type}"
    
    if model_type == None:
        name += "_(m_t)_None"
    else:
        name += f"_(m_t)_{model_type}"
    
    count=0
    temp = ""
    while Utils.check_folder_exists(folder_path=project_path / f"{name}{temp}"):
        # print(F"<--- folder exists --->")
        count +=1
        temp = f"_{count}"
        
    return f"{name}{temp}"

if __name__ == "__main__":
    print(f"--- Initial Train Parameters ---")
    # TODO: Initial parameters
    project_name = Constants.experiment_path

    # create experiment folder
    if not Utils.check_folder_exists(project_name):
        Utils.delete_folder_mkdir(folder_path=project_name, remove=False)

    experiment_name = experiment_name(dataset_type=dataset_dict["DATASET_TYPE"], model_type=model_dict["MODEL_TYPE"], project_path=project_name)

    print("-" * 100)
    print(project_name)
    print("-" * 100)

    train_parameters = TrainParameters(
        gauge_type=dataset_dict["DATASET_TYPE"],
        model_type =model_dict["MODEL_TYPE"], 
        data_yaml_path=dataset_dict["DATASET_PATH"],
        epochs=train_parameters_dict["EPOCHS"],
        imgsz=train_parameters_dict["IMG_SIZE"],
        batch=train_parameters_dict["BATCH_SIZE"],
        cache=train_parameters_dict["CACHE"],
        patience=train_parameters_dict["PATIENCE"],
        device=train_parameters_dict["DEVICE"],
        workers=train_parameters_dict["WORKERS"],
        resume=train_parameters_dict["RESUME"],
        learning_rate=train_parameters_dict["LEARNING_RATE"],
        final_learning_rate=train_parameters_dict["FINAL_LEARNING_RATE"],
        project_name=project_name.name,
        name=experiment_name,
    )

    print(f"--- Prepare Data ---")
    dataset = YOLODataset(dataset_type=dataset_dict["DATASET_TYPE"], dataset_path=dataset_dict["DATASET_PATH"])
    train_parameters.data_yaml_path = dataset.get_data_yaml4train()
    # ic(dataset.get_data_yaml4train(), train_parameters.data_yaml_path)

    # print(len(dataset))
    # dataset.show_samples(number_of_samples=5,random_seed=21)

    print(f"---- Loading Model ---")

    yolo_model = YOLOModel(
        model_type=Utils.get_enum_by_value(value=model_dict["MODEL_TYPE"].upper(), enum=Constants.ModelType),
        use_comet=True,
        gauge_type=dataset_dict["DATASET_TYPE"],
        pretrain_path=None,
    )


    print(f"--- Training Model ---")
    ic(train_parameters.comet_parameters())
    yolo_model.train(parameters=train_parameters)

    #     print(f"--- Export Model ---")


    #     print(f"--- Validate Model ---")


    #     print(f"--- Testing Model ---")
