import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class InferenceUtils:
    @staticmethod
    def is_overlapping(bbox1, bbox2):
        """
        Checks if two bounding boxes overlap.

        Args:
            bbox1 (list): The first bounding box, as a list of four coordinates (x1, y1, x2, y2).
            bbox2 (list): The second bounding box, as a list of four coordinates (x1, y1, x2, y2).

        Returns:
            bool: True if the two bounding boxes overlap, False otherwise.
        """
        print(f"bbox1: {bbox1}, bbox2: {bbox2}")
        x1, y1, x2, y2 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        if x2 < x1_2 or x2_2 < x1:
            return False
        if y2 < y1_2 or y2_2 < y1:
            return False

        return True

    @staticmethod
    def albu_resize_pad_zero(target_size, format=None):
        from utils.constants import Constants

        target_width = target_size[0]
        target_height = target_size[1]

        if (format == None) or (format == Constants.BoundingBoxFormat.YOLOV8):
            transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=max(target_size)),
                    A.PadIfNeeded(
                        min_height=target_height,
                        min_width=target_width,
                        border_mode=cv2.BORDER_CONSTANT,
                        
                    ),
                    A.Resize(
                        height=target_height, width=target_width, always_apply=True
                    ),
                    ToTensorV2(),
                ],
            )

        return transform

    @staticmethod
    def albu_make_frame(img, frame_coor,target_size):
        target_width = img.shape[0] -1
        target_height = img.shape[1] -1

        target_resize_width = target_size[0]
        target_resize_height = target_size[1]


        transform = A.Compose(
            [
                A.Crop(
                    x_min=max(0, frame_coor[0] ),
                    y_min=max(0, frame_coor[1]),
                    x_max=min(target_width, frame_coor[2]),
                    y_max=min(target_height, frame_coor[3]),
                    
                ),
                A.LongestMaxSize(max_size=max(target_size)),
                A.PadIfNeeded(
                    min_height=target_size[0],
                    min_width=target_size[1],
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.Resize(
                    height=target_resize_height, width=target_resize_width, always_apply=True
                ),
                ToTensorV2(),
            ],
        ) 
        return transform