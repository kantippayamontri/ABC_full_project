from utils import Constants, Utils
from train import YOLODataset, YOLOModel, TrainParameters
import argparse
from icecream import ic

# # Check arguments
# argParser = argparse.ArgumentParser()
# argParser.add_argument("-gt","--gauge_type", help="gauge_type")
# argParser.add_argument("-mt","--model_type", help="model_type")
# argParser.add_argument("-ep","--epochs", help="epochs")
# argParser.add_argument("-imgs","--image_size", help="image_size")
# argParser.add_argument("-bs","--batch_size", help="batch size")
# argParser.add_argument("-c","--cache", help="cache")
# argParser.add_argument("-p","--patience", help="patience")
# argParser.add_argument("-d","--device", help="device")
# argParser.add_argument("-w","--workers", help="workers")
# argParser.add_argument("-","--", help="")
# argParser.add_argument("-","--", help="")
# argParser.add_argument("-","--", help="")

# args = argParser.parse_args()
# print("args=%s" % args)
# print("args.gauge_type=%s" % args.gauge_type)

def experiment_name(dataset_type=None, model_type=None, project_path=None):
    name = ""
    if dataset_type == None:
        name += "(ds_t)_None"
    else:
        name += f"(ds_t)_{dataset_type.value}"
    
    if model_type == None:
        name += "_(m_t)_None"
    else:
        name += f"_(m_t)_{model_type.value}"
    
    count=0
    temp = ""
    while Utils.check_folder_exists(folder_path=project_path / f"{name}{temp}"):
        print(F"<--- folder exists --->")
        count +=1
        temp = f"_{count}"
        
    return f"{name}{temp}"
    

print(f"--- Initial Train Parameters ---")
# TODO: Initial parameters
dataset_type = Constants.GaugeType.digital
model_type = Constants.ModelType.NANO
project_name = ic(Constants.experiment_path)
# create experiment folder
if not Utils.check_folder_exists(project_name):
    Utils.delete_folder_mkdir(folder_path=project_name, remove=False)

experiment_name = ic(experiment_name(dataset_type=dataset_type, model_type=model_type, project_path=project_name))

train_parameters = TrainParameters(
    gauge_type=dataset_type,
    model_type = model_type,
    data_yaml_path=None,
    epochs=2,
    imgsz=1024,
    batch=8,
    cache=True,
    patience=20,
    device="mps",
    workers=8,
    resume=True,
    learning_rate=0.001,
    final_learning_rate=0.01,
    project_name=project_name,
    name=experiment_name
)

print(f"--- Prepare Data ---")
dataset = YOLODataset(dataset_type=dataset_type)
train_parameters.data_yaml_path = dataset.get_data_yaml4train()
print(len(dataset))
# dataset.show_samples(number_of_samples=5,random_seed=21)

print(f"---- Loading Model ---")

yolo_model = YOLOModel(
    model_type=model_type,
    use_comet=True,
    gauge_type=dataset_type
)


print(f"--- Training Model ---")
yolo_model.train(parameters=train_parameters)

print(f"--- Export Model ---")


print(f"--- Validate Model ---")


print(f"--- Testing Model ---")
