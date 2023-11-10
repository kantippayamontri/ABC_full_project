import preprocess
from utils import Constants

datasetCombineModel = preprocess.DatasetCombineModel(make_frame=False)
# datasetCombineModel.conduct_dataset(delete_dataset_for_train=True)
datasetCombineModel.visualize_samples(gauge_type=Constants.GaugeType.digital,number_of_samples=30)

