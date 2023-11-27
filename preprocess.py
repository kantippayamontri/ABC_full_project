import preprocess
from utils import Constants

datasetCombineModel = preprocess.DatasetCombineModel(make_frame=False, dataset_choose=Constants.GaugeType.number)
# datasetCombineModel.conduct_dataset(delete_dataset_for_train=True)
datasetCombineModel.visualize_samples(gauge_type=Constants.GaugeType.number,number_of_samples=10)

