from utils import Constants, Utils
from train import YOLODataset, YOLOModel, TrainParameters
import argparse
from icecream import ic

# Create ArgumentParser object
parser = argparse.ArgumentParser(description="for training arguments parser")

# Add positional argument
parser.add_argument("input_file",type=str, help = "your python file to run")
parser.add_argument("dataset_type", type=str, help="type of dataset to train")


# Add optional argument with a default value
parser.add_argument("-ep", "--epochs", type=int, default=300, help="a number of epochs")
parser.add_argument("-imgs", "--img_size", type=int, choices=[640,1024], default=1024, help="size of train images")
parser.add_argument("-bs", "--batch_size", type=int, default=16, help="number of batch size")
parser.add_argument("-c", "--cache",type=bool, default=False, help="cache the image for True and False")
parser.add_argument("-p", "--patience", type=int, default=20, help="set number of how many not learning epochs to stop training")
parser.add_argument("-d", "--device",type=str,choices=["cpu","mps","0","1"], default="cpu", help="choose device to train")
parser.add_argument("-w", "--workers", type=int, default=12 , help="set the number of workers")
parser.add_argument("-rs", "--resume", type=bool, default=False, help="resume training")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("-flr", "--final_learning_rate", type=float,default=0.01, help="final learning rate")

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
input_file = args.input_file
dataset_type = Utils.get_enum_by_value(value=args.dataset_type.lower(),enum=Constants.GaugeType)
model_type = Utils.get_enum_by_value(value=args.model_type.upper(), enum=Constants.ModelType)
epochs = args.epochs
image_size = args.img_size
batch_size = args.batch_size
cache = args.cache
patience = args.patience
device = args.device
workers = args.workers
resume = args.resume
learning_rate = args.learning_rate
final_learning_rate = args.final_learning_rate

# Your program logic goes here
ic(f"Input file: {input_file}")
ic(f"Dataset type: {dataset_type.value}")
ic(f"Model type: {model_type.value}")
ic(f"Epochs: {epochs}")
ic(f"Image size: {image_size}")
ic(f"Batch size: {batch_size}")
ic(f"Cache: {cache}")
ic(f"Patience: {patience}")
ic(f"Device: {device}")
ic(f"Workers: {workers}")
ic(f"Resume: {resume}")
ic(f"Learning Rate: {learning_rate}")
ic(f"Final learning rate: {final_learning_rate}")

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
# dataset_type = Constants.GaugeType.digital
# model_type = Constants.ModelType.NANO
project_name = ic(Constants.experiment_path)
# create experiment folder
if not Utils.check_folder_exists(project_name):
    Utils.delete_folder_mkdir(folder_path=project_name, remove=False)

experiment_name = ic(experiment_name(dataset_type=dataset_type, model_type=model_type, project_path=project_name))

train_parameters = TrainParameters(
    gauge_type=dataset_type,
    model_type = model_type,
    data_yaml_path=None,
<<<<<<< HEAD
    epochs=epochs,
    imgsz=image_size,
    batch=batch_size,
    cache=cache,
    patience=patience,
    device=device,
    workers=workers,
    resume=resume,
    learning_rate=learning_rate,
    final_learning_rate=final_learning_rate,
=======
    epochs=300,
    imgsz=1024,
    batch=36,
    cache=True,
    device="0",
    workers=24,
    resume=True,
    learning_rate=0.001,
    final_learning_rate=0.01,
>>>>>>> f367bae (add some file for train in erawn)
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
