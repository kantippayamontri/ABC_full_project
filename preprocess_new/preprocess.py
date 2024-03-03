from icecream import ic
import typing
from utils import Utils


class Preprocess:
    def __init__(self, preprocess_dict, dataset_path):
        self.preprocess_dict = preprocess_dict
        self.target_folder = dataset_path

        ic(self.preprocess_dict,self.target_folder)

        for folder in self.preprocess_dict["FOLDER"]: #FIXME: choose this line instead
        # for folder in ["valid"]:
            print(f'\t[-] PREPROCESS AT {str(dataset_path / folder)}')
            self.preprocess(
                preprocess_list=self.preprocess_dict["PREPROCESS_LIST"],
                dataset_folder=dataset_path / folder,
            )

    def preprocess(self, preprocess_list=[], dataset_folder=None):
        from .transforms import Transform

        # FIXME: uncomment this for
        if preprocess_list == []:
            print(f"\t\t[X] NO NEED TO PREPROCESS -> PREPROCESS LIST = 0")
            return

        if dataset_folder is None:
            print(f"\t\t[X] PREPROCESS FAIL -> NO DATASET FOLDER")
            return


        for pre_d in preprocess_list:

            matches_img_bb = Utils.get_filename_bb_folder(
                img_path=dataset_folder / "images", bb_path=dataset_folder / "labels"
            )
            (function_name, function_parameter) = tuple(
                (key, value) for key, value in pre_d.items()
            )[0]

            print(f"\t\t[-] {function_name}")

            for _, (img_path, bb_path) in enumerate(matches_img_bb):
                img = Utils.load_img_cv2(filepath=img_path)
                bb = Utils.load_bb(filepath=bb_path)

                # if _ > 0:
                #     break

                if (img is None) or (bb is None):
                    continue

                transform = Transform(img_path=img_path, bb_path=bb_path)
                img, bb = transform.transform_dict_function(
                    function_name=function_name,
                    function_parameter=function_parameter,
                    img=img.copy(),
                    bb=bb.copy(),
                    target_folder_path=dataset_folder
                )
                
            print(f"\t\t\t[/] {function_name} success.")
        
        Utils.visualize_samples(source_folder=dataset_folder, number_of_samples=10 , gauge_type="digital")
            

        
