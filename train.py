from utils import Constants
from train import YOLODataset, YOLOModel, TrainParameters

print(f"--- Training Data ---")
# print(Constants.DatasetUse.TYPE_3_GAUGE_DISPLAY_FRAME.value)
dataset = YOLODataset(
    dataset_dict=Constants.dataset_digital[
        Constants.DatasetUse.TYPE_3_GAUGE_DISPLAY_FRAME.value
    ],
    remove_exists=True,
)
# dataset.change_settings_dataset()

# print(f"dataset train : {dataset.dataset_train_path}")
dataset.show_samples(
    number_of_samples=15, datasetTrainValTest=Constants.DatasetTrainValTest.VAL
)
yolo_model = YOLOModel(
    model_type=Constants.ModelType.NANO,
    use_comet=True,
    dataset_type=Constants.DatasetType.TYPE_3,
    dataset_use=Constants.DatasetUse.TYPE_3_GAUGE_DISPLAY_FRAME,
)
print(f"data yaml path: {dataset.get_data_yaml4train()}")
yolo_model.train(
    parameters=TrainParameters(
        data_yaml_path=dataset.get_data_yaml4train(),
        epochs=100,
        imgsz=[1280, 1280],
        batch=4,
        cache=False,
        patience=25,
        device="mps",
        workers=4,
        resume=True,
    )
)
