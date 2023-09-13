from roboflow import Roboflow
from utils.utils import Utils
from utils.constants import Constants
import pathlib
import os
import cv2
import numpy as np
import yaml


class DatasetFromRoboflow:
    def __init__(
        self,
        version=None,
        api_key=None,
        project_name=None,
        model_format=None,
        dataset_folder=None,
        type=None,
        key=None,
        remove_exist=True,
    ):
        self.version = version
        self.api_key = api_key
        self.project_name = project_name
        self.model_format = model_format
        self.dataset_folder = dataset_folder
        self.remove_exist = remove_exist
        self.key = key
        self.type = type

    def import_datasets(
        self,
    ):
        if Utils.delete_folder_mkdir(self.dataset_folder, remove=self.remove_exist):

            # download dataset from roboflow
            rf = Roboflow(
                api_key=self.api_key,
                model_format=self.model_format,
            )
            rf.workspace().project(self.project_name).version(self.version).download(
                location=str(self.dataset_folder)
            )
            # self.preprocess()
        else:
            print(f"--- folder exists not download dataset ---")
    
    
    def parepare_for_train(self, ):
        return

    def preprocess(self):
        key = self.key
        type = self.type
        # print(f"preprocess, key: {key}, type: {type}")
        target_class_map_dict = self.map_class_yaml(
            yaml_path=self.dataset_folder / "data.yaml",
            map_dict=Constants.map_data_dict[type][key],
        )
        # return
        for _ in [
            Constants.train_folder,
            Constants.val_folder,
        ]:  # train and valid folder
            # for _ in [Constants.val_folder]: # train and valid folder
            image_filenames = os.listdir(
                self.dataset_folder / _ / Constants.image_folder
            )
            for count, image_filename in enumerate(
                image_filenames
            ):  # loop filenames and labelnames
                # print(f"image filename: {image_filename}")
                image_path = (
                    self.dataset_folder / _ / Constants.image_folder / image_filename
                )
                label_path = (
                    self.dataset_folder
                    / _
                    / Constants.label_folder
                    / (image_filename[:-3] + "txt")
                )

                if (count + 1) % 300 == 0:
                    print(f"image {count + 1} succcess")

                if image_path.is_file() and label_path.is_file():
                    # load image
                    img = cv2.imread(str(image_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # load bounding box
                    bb = []

                    fp = open(label_path, "r")  # read the bounding box
                    for c, line in enumerate(fp):
                        bb.append(list(float(n) for n in line.split(" ")))

                    bb = np.array(bb)

                    if type == "type3":
                                
                        if key == "number":
                            
                            new_img_bb = self.preprocess_number(
                                img=img, bb=bb, map_dict=target_class_map_dict
                            )
                            
                            for index, _n in enumerate(new_img_bb):
                                # Utils.visualize_img_bb(img=new_img_bb[index][0], bb=new_img_bb[index][1],with_class=True,labels=list(target_class_map_dict['target'].keys()))
                                new_label_path = Utils.change_filename_sample(filepath=label_path,filename=image_filename,index=index,extension='.txt', start_index=0)
                                new_img_path = Utils.change_filename_sample(filepath=image_path, filename=image_filename, index=index, start_index=0)
                                Utils.save_image(img=new_img_bb[index][0], filepath=new_img_path)
                                self.overwrite_label(new_label_path, new_img_bb[index][1])
                                # if count == 10: return
                            
                        elif key == "gauge_display_frame":
                            
                            new_img_bb = self.preprocess_gauge_display_frame(img=img, bb=bb, map_dict=target_class_map_dict)
                            
                            for index, _n in enumerate(new_img_bb):
                                # Utils.visualize_img_bb(img=new_img_bb[index][0], bb=new_img_bb[index][1],with_class=True,labels=list(target_class_map_dict['target'].keys()))
                                new_label_path = Utils.change_filename_sample(filepath=label_path,filename=image_filename,index=index,extension='.txt', start_index=0)
                                new_img_path = Utils.change_filename_sample(filepath=image_path, filename=image_filename, index=index, start_index=0)
                                Utils.save_image(img=new_img_bb[index][0], filepath=new_img_path)
                                self.overwrite_label(new_label_path, new_img_bb[index][1])

                    

        if type == "type3":
            if key == "gauge":
                self.change_data_yaml(
                    data_yaml_path=self.dataset_folder / "data.yaml",
                    target_dict=target_class_map_dict["target"],
                )
            elif key == "frame":
                self.change_data_yaml(
                    data_yaml_path=self.dataset_folder / "data.yaml",
                    target_dict=target_class_map_dict["final_target"],
                )
            elif key == "number":
                self.change_data_yaml(
                    data_yaml_path=self.dataset_folder / "data.yaml",
                    target_dict=target_class_map_dict["target"],
                    plus=1
                )
            elif key == "gauge_display_frame":
                self.change_data_yaml(
                    data_yaml_path=self.dataset_folder / "data.yaml",
                    target_dict=target_class_map_dict["target"],
                )

    def map_class_yaml(self, yaml_path=None, map_dict=None):
        yaml_dict = Utils.read_yaml_file(yaml_path)
        source = map_dict["source"]
        target = map_dict["target"]
        final_target = None
        
        if "final_target" in map_dict:
            final_target = map_dict["final_target"]
            

        yaml_classes = yaml_dict["names"]

        result_source = {}
        result_target = {}
        result_final_target = {}

        for index, c in enumerate(source):
            try:
                result_source[c] = yaml_classes.index(c)
            except:
                result_source[c] = 99
        result_source = {key: value for key, value in sorted(result_source.items(), key=lambda item: item[1])}


        for index, c in enumerate(target):
            result_target[c] = index
        result_target = {key: value for key, value in sorted(result_target.items(), key=lambda item: item[1])}
        
        if final_target != None:
            print(f"map dict: {map_dict}")
            print(f"result target: {result_target}")
            
            for index, k in enumerate(final_target):
                result_final_target[k] = result_target[k]
            
            sorted_final_target_dict = dict(sorted(result_final_target.items(), key=lambda item: item[1]))
            for index, k in enumerate(sorted_final_target_dict.keys()):
                sorted_final_target_dict[k] = index
            
            return {"source": result_source, "target": result_target, "final_target": sorted_final_target_dict}
    
        return {"source": result_source, "target": result_target, "final_target": None}
    
    def preprocess_gauge_display_frame(self,img=None, bb=None, map_dict=None):
        new_img_bb_list = []
        
        new_img_bb = Utils.resize_img_bb(target_size=[1280,1280], img=img, bb=bb) #resize the image
        new_img_bb[1] = Utils.reclass_bb_from_dict(bb=new_img_bb[1],bb_dict_before=map_dict['source'], bb_dict_after=map_dict['target']) #reclass image
        
        new_img_bb_list.append(new_img_bb)
        
        return new_img_bb_list

    def preprocess_gauge(self, img, bb, class_dict={}, map_dict=None):
        source_gauge = map_dict["source"]["gauge"]
        source_display = map_dict["source"]["display"]
        source_frame = map_dict["source"]["frame"]

        target_gauge = map_dict["target"]["gauge"]
        target_display = map_dict["target"]["display"]

        # check display exist
        if source_display not in bb[:, 0]:  # if display doesn't exist
            for index in range(len(bb)):
                if int(bb[index][0]) == source_frame:  # change frame to display
                    bb[index][0] = source_display

        # remove frame class
        bb_temp = []
        for _bb in bb:
            if int(_bb[0]) != source_frame:
                bb_temp.append(_bb)
        bb = np.array(bb_temp)

        # reclass
        for index in range(len(bb)):
            if int(bb[index][0]) == source_gauge:  # change gauge to 1
                bb[index][0] = target_gauge
            elif int(bb[index][0]) == source_display:
                bb[index][0] = target_display

        return img, bb

    def preprocess_frame(self, img=None, bb=None, map_dict=None, ):
        # print(f"preprocess frame, map_dict={map_dict}")
        source_frame = map_dict["source"]["frame"]
        source_display = map_dict["source"]["display"]

        target_frame = map_dict["target"]["frame"]
        target_display = map_dict["target"]["display"]

        # check if display exists
        bb_create = []
        if source_display not in bb[:, 0]:
            # print(f"not found display")
            for index in range(len(bb)):
                if int(bb[index][0]) == source_frame:
                    bb_create.append(
                        [
                            source_frame,
                            bb[index][1],
                            bb[index][2],
                            bb[index][3] - 0.002,
                            bb[index][4] - 0.002,
                        ]
                    )
                    bb[index][0] = source_display

        if len(bb_create) > 0:
            bb = np.append(bb, np.array(bb_create), axis=0)

        # reclass
        result_bb = []
        for index in range(len(bb)):
            if int(bb[index][0]) == source_frame:
                bb[index][0] = target_frame
                result_bb.append(bb[index])
            elif int(bb[index][0]) == source_display:
                bb[index][0] = target_display
                result_bb.append(bb[index])
        return img, result_bb

    def preprocess_number(self, img, bb, map_dict=None):
        new_img_bb_list = []
        
        new_img_bb = Utils.make_crop_image_and_bb(
                                img=img,
                                bb=bb,
                                class_crop=map_dict["source"]["frame"],
                                labels = dict(sorted(map_dict['source'].items(), key=lambda item: item[1])),
                                crop_target_size = [1280,1280]
                            )
        
        for index, (_img, _bb) in enumerate(new_img_bb):
            new_img_bb[index] = list(new_img_bb[index])
            new_img_bb[index][1] = Utils.reclass_bb_from_dict(bb=new_img_bb[index][1], bb_dict_before=map_dict['source'], bb_dict_after=map_dict['target'],class_crop=map_dict["source"]["frame"])
            new_img_bb_list.append(new_img_bb[index])
        
        return new_img_bb_list

    def overwrite_label(self, txt_file_path, bb):
        file = open(txt_file_path, "w")
        # Write new content to the file
        str_save = ""
        for _bb in bb:
            str_save += f"{int(_bb[0])} {_bb[1]} {_bb[2]} {_bb[3]} {_bb[4]}\n"
        file.write(str_save)
        file.close()

    def change_data_yaml(self, data_yaml_path=None, target_dict=None,plus=None):
        sorted_target_dict = dict(sorted(target_dict.items(), key=lambda item: item[1]))

        f = open(data_yaml_path, "r")
        y = yaml.safe_load(f)
        
        names = list(sorted_target_dict.keys())
        nc = len(sorted_target_dict.keys())
        
        if plus != None:
            for index in range(plus):
                nc += 1
                names.append(f"dummy_{index+1}")
            
        y["names"] = names
        y["nc"] = nc
        outfile = open(data_yaml_path, "w")
        yaml.dump(y, outfile, default_flow_style=False)

    