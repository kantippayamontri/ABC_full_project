# ABC_full_project

For preprocess
-> Prepare dataset 1. put the datset folder in folder ./datasets/{name dataset}
ex. ./datasets/digital/ 2. run preprocess.py
ex. python preprocess.py 3. the dataset will appear in datasets_for_train/{name dataset}
ex. ./datasets_for_train/digital/

-> command to train

    python train.py requirements.txt digital SMALL --epochs 100 --img_size 1024 --batch_size 32 --cache True --patience 15 --device cpu --workers 20 --resume True -lr 0.001

-> predict.py
python predict.py --input_file=requirements.txt --gauge_use=digital --model_path="models/digital/digital_model.pt" --img_path="datasets_for_train/digital/test/images/" --bb_path="datasets_for_train/digital/test/labels/" --image_size=640 --conf=0.25

-> inference.py
python inference.py --input_file=requirements.txt --gauge_use=digital --img_path="./test_image/digital/" 
-> inference -> for select frames
python inference.py --input_file=requirements.txt --gauge_use=number --img_path="./datasets/number/number_test1/train/images" --select_frame=True   

-> val.py
python val.py requirements.txt digital ./models/digital/digital_model.pt ./datasets_for_train/digital/test/ --plot False
