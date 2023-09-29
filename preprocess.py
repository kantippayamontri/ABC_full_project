
# for k,v in Constants.dataset_digital.items(): 
#     dataset_dict = Constants.dataset_digital[k]
    # print(f"dataset_dict : {dataset_dict}")
    # data = DatasetFromRoboflow(version=dataset_dict["dataset dict"]['version'],
    #                         api_key=dataset_dict["dataset dict"]['api_key'],
    #                         project_name=dataset_dict["dataset dict"]['project_name'],
    #                         model_format=dataset_dict["dataset dict"]['model_format'],
    #                         dataset_folder=dataset_dict["dataset dict"]['dataset_folder'],
    #                         key = dataset_dict['key'],
    #                         type = dataset_dict['type'],
    #                         remove_exist=True,
    #                         parameters=dataset_dict['parameters'],
    #                         )
    # data.import_datasets()
    # data.preprocess()
    # break


# #TODO: step 1: import dataset

# # print(f"--- PASSED ---")
# # for k,v in Constants.dataset_digital.items(): 
# #     # dataset_dict = Constants.dataset_digital[k]
# #     # print(dataset_dict)
# #     print(f"key: {k}, type: {v}")
import preprocess
    
datasetBaseModel = preprocess.DatasetBaseModel()
datasetBaseModel.conduct_dataset()


# #TODO: step 2: combine multiple datasets

# #TODO: step 3: preprocess dataset
