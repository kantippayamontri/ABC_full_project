from utils import Constants
from train import YOLODataset, YOLOModel, TrainParameters

print(f"--- Initial Train Parameters ---")
# TODO: Initial parameters
dataset_type = Constants.GaugeType.digital
model_type = Constants.ModelType.NANO
train_parameters = TrainParameters(
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
    # learning_rate=0.001,
    # final_learning_rate=0.001,
)

print(f"--- Prepare Data ---")
dataset = YOLODataset(dataset_type=dataset_type)
train_parameters.data_yaml_path = dataset.get_data_yaml4train()
print(len(dataset))
# dataset.show_samples(number_of_samples=5,random_seed=29)

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

# dataset.show_samples(
#     number_of_samples=15, datasetTrainValTest=Constants.DatasetTrainValTest.VAL
# )

# yolo_model = YOLOModel(
#     model_type=Constants.ModelType.NANO,
#     use_comet=True,
#     dataset_type=Constants.DatasetType.TYPE_3,
#     dataset_use=Constants.DatasetUse.TYPE_3_GAUGE_DISPLAY_FRAME,
# )

# print(f"data yaml path: {dataset.get_data_yaml4train()}")
# yolo_model.train(
#     parameters=TrainParameters(
#         data_yaml_path=dataset.get_data_yaml4train(),
#         epochs=100,
#         imgsz=[1280, 1280],
#         batch=4,
#         cache=False,
#         patience=25,
#         device="mps",
#         workers=4,
#         resume=True,
#     )
# )
