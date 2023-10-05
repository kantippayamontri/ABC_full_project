from utils import Constants, Utils


class ProprocessGaugeModel:
    def __init__(
        self,
        match_img_bb_path=None,
        gauge_type=None,
        source_folder=None,
    ):
        self.match_img_bb_path = match_img_bb_path
        self.gauge_type = gauge_type
        self.source_folder = source_folder
        return

    def preprocess(
        self,
    ):
        from utils import Constants

        if self.gauge_type is None or self.gauge_type == "":
            print(f"\t\t[X] PLEASE SPECIFIC GAUGE TYPE")
            return

        if self.match_img_bb_path == None:
            print(f"\t\t[X] PLEASE PASS IMAGE PATH AND LABELS PATH")
            return

        if self.gauge_type == Constants.GaugeType.digital.value:
            self.preprocess_digital(gauge_name=Constants.GaugeType.digital.value)

        if self.gauge_type == Constants.GaugeType.dial.value:
            self.preprocess_dial(gauge_name = Constants.GaugeType.dial.value)

        if self.gauge_type == Constants.GaugeType.number.value:
            self.preprocess_number(gauge_name = Constants.GaugeType.number.value)

        if self.gauge_type == Constants.GaugeType.level.value:
            self.preprocess_level(gauge_name = Constants.GaugeType.level.value)

        if self.gauge_type == Constants.GaugeType.clock.value:
            self.preprocess_clock(gauge_name = Constants.GaugeType.clock.value)

        return

    def preprocess_digital(self, gauge_name=None):
        print(f"\t\t[-] PREPROCESS DIGITAL")
        # TODO: add images that crop display and gauge
        number_crop_gauge = 0
        number_crop_display = 0
        for index, (img_path, bb_path) in enumerate(self.match_img_bb_path):
            print(f"image index: {index}")
            # if index > 5:
            #     return

            img = Utils.load_img_cv2(filepath=img_path)
            bb = Utils.load_bb(filepath=bb_path)

            target_dict = {
                value: index
                for index, value in enumerate(
                    Constants.map_data_dict[Constants.GaugeType.digital.value]["target"]
                )
            }
            target_size = [img.shape[0], img.shape[1]]
            labels = list(target_dict.keys())
            img_bb_crop_gauge = []
            img_bb_crop_display = []

            # TODO: crop gauge class
            img_bb_crop_gauge = Utils.crop_img(
                img=img,
                bb=bb,
                class_crop=target_dict["gauge"],
                need_resize=True,
                target_size=target_size,
            )
            # TODO: save image and label crop gauge
            for index, (_img, _bb) in enumerate(img_bb_crop_gauge):
                new_name = Utils.new_name_crop_with_date(
                    gauge=gauge_name,
                    number="crop",
                    class_crop="gauge",
                    image_number=number_crop_gauge,
                )
                
                number_crop_gauge += 1
                
                # print(f"new name: {new_name}")
                # TODO: save image
                new_name_path_image = Utils.change_file_name(old_file_name=img_path, new_name=new_name, isrename=False)
                Utils.save_image(img=_img, filepath=new_name_path_image)
                # print(f"new_name_path_image: {new_name_path_image}")
                # TODO: save label
                new_name_path_label = Utils.change_file_name(old_file_name=bb_path, new_name=new_name, isrename=False)
                Utils.overwrite_label(txt_file_path=new_name_path_label, bb=_bb)
                # print(f"new_name_path_label: {new_name_path_label}")
                
                
            # TODO: crop display class
            img_bb_crop_display = Utils.crop_img(
                img=img,
                bb=bb,
                class_crop=target_dict["display"],
                need_resize=True,
                target_size=target_size,
            )
            # TODO: save image and label crop display
            for index, (_img, _bb) in enumerate(img_bb_crop_display):
                new_name = Utils.new_name_crop_with_date(
                    gauge=gauge_name,
                    number="crop",
                    class_crop="display",
                    image_number=number_crop_display,
                )
                
                number_crop_display += 1
                
                # TODO: save image
                # print(f"new name: {new_name}")
                new_name_path_image = Utils.change_file_name(old_file_name=img_path, new_name=new_name,isrename=False)
                Utils.save_image(img=_img, filepath=new_name_path_image)
                # print(f"new_name_path_image: {new_name_path_image}")
                
                # TODO: save label
                new_name_path_label = Utils.change_file_name(old_file_name=bb_path, new_name=new_name,isrename=False)
                Utils.overwrite_label(txt_file_path=new_name_path_label, bb=_bb)
                # print(f"new_name_path_label: {new_name_path_label}")


    def preprocess_dial(
        self,
    ):
        print(f"\t\t[-] PREPROCESS DIAL")
        return

    def preprocess_number(
        self,
    ):
        print(f"\t\t[-] PREPROCESS NUMBER")
        return

    def preprocess_level(
        self,
    ):
        print(f"\t\t[-] PREPROCESS LEVEL")
        return

    def preprocess_clock(
        self,
    ):
        print(f"\t\t[-] PREPROCESS CLOCK")
        return
