from preprocess.dataset_base_model import DatasetBaseModel
from preprocess.preprocess_constants import PreprocessConstants

if __name__ == '__main__':
    #TODO: step 1: import dataset
    datasetBaseModel = DatasetBaseModel() # * for combine multiple datasets into on folder in folder ./dataset_for_train
    datasetBaseModel.conduct_dataset()


    #TODO: step 2: combine multiple datasets

    #TODO: step 3: preprocess dataset