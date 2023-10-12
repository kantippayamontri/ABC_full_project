# ABC_full_project

For preprocess
-> Prepare dataset
    1. put the datset folder in folder ./datasets/{name dataset}
        ex. ./datasets/digital/
    2. run preprocess.py 
        ex. python preprocess.py
    3. the dataset will appear in datasets_for_train/{name dataset}
        ex. ./datasets_for_train/digital/

-> commend to train
python train.py -gt="digital" 