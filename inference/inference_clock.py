import math

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from utils import Utils


class InferenceClock:
    def __init__(
        self,
        case="normal",
        gauge=None,
        min=None,
        max=None,
        center=None,
        head=None,
        bottom=None,
        needle=None,
        min_value=0,
        max_value=100,
    ) -> None:
        self.case = case
        self.gauge = gauge
        self.min = min
        self.max = max
        self.center = center
        self.head = head
        self.bottom = bottom
        self.needle = needle
        self.min_value = min_value
        self.max_value = max_value
        self.min_max_swap = False

        if self.min is not None and self.max is not None:
            self.case = self.check_circle_type()

        self.preprocess_clock()
        self.min_max_swap = self.check_min_max_swap()
        # self.visualize_clock()
    
    def check_circle_type(self,):
        # check min and max is overlap
        min_box = [(self.min[0], self.min[1]), (self.min[2], self.min[3])]
        max_box = [(self.max[0], self.max[1]), (self.max[2], self.max[3])]
        if Utils.calculate_intersection(box1=min_box, box2=max_box):
            return "circle"

        return "normal"

    def check_min_max_swap(
        self,
    ) -> bool:
        if not ((self.min is not None) or (self.max is not None)):
            return False

        _max_center = self.get_center_point(point=self.max)
        _min_center = self.get_center_point(point=self.min)

        if _max_center[0] < _min_center[0]:  # check x position of min and max
            self.min, self.max = self.max, self.min
            ic(f"need to swap min and max")
            return True

        return False

    def preprocess_clock(
        self,
    ):
        if self.case == "normal":
            self.preprocess_normal()
        elif self.case == "part":
            self.preprocess_part()
        elif self.case == "circle":
            self.preprocess_circle()

    def preprocess_normal(
        self,
    ):
        for _ in range(1):
            if self.center is None:  # not found center
                if (self.bottom is not None) and (
                    self.head is not None
                ):  # found bottom and head
                    _bottom_point = self.get_center_point(self.bottom)
                    _head_point = self.get_center_point(self.head)

                    # make the average 2 round -> bottom is nearer center than head
                    _avg_bottom_head = self.get_center_point(
                        np.array(
                            [
                                _bottom_point[0],
                                _bottom_point[1],
                                _head_point[0],
                                _head_point[1],
                            ]
                        )
                    )
                    _avg_bottom_head = self.get_center_point(
                        np.array(
                            [
                                _bottom_point[0],
                                _bottom_point[1],
                                _avg_bottom_head[0],
                                _avg_bottom_head[1],
                            ]
                        )
                    )

                    self.center = np.array(
                        [
                            _avg_bottom_head[0],
                            _avg_bottom_head[1],
                            _avg_bottom_head[0],
                            _avg_bottom_head[1],
                        ]
                    )
                elif (self.head is not None) and (
                    self.needle is not None
                ):  # foud head and needle
                    # center is point(from needle) that farest from head
                    _center_head = self.get_center_point(point=self.head)
                    n_x_min, n_y_min, n_x_max, n_y_max = self.needle
                    # find 4 point of needle
                    n_tl = (n_x_min, n_y_min)  # top left point
                    n_tr = (n_x_max, n_y_min)  # top righ
                    n_bl = (n_x_min, n_y_max)  # bottom left point
                    n_br = (n_x_max, n_y_max)  # bottom righ point

                    needle_point = [n_tl, n_tr, n_bl, n_br]  # list of needle point
                    needle_point_dict = dict(
                        zip(
                            needle_point,
                            [
                                self.get_distance_from_2_point(
                                    point1=_center_head, point2=_p
                                )
                                for _p in needle_point
                            ],
                        )
                    )  # dict key=point , value=distance
                    needle_max_dis_point = max(
                        needle_point_dict, key=lambda k: needle_point_dict[k]
                    )

                    self.center = np.array(
                        [
                            needle_max_dis_point[0],
                            needle_max_dis_point[1],
                            needle_max_dis_point[0],
                            needle_max_dis_point[1],
                        ]
                    )

                    # ic(needle_point)
                    # ic(needle_point_dict)
                    # ic(needle_max_dis_point)
                    # max_dis_point = needle_point[0]
                    # max_dis=0
                    # for _needle_p in needle_point:
                    #     _dis = self.get_distance_from_2_point(point1=_center_head , point2=_needle_p)
                    #     if _dis > max_dis:

            if self.head is None:
                if (self.bottom is not None) and (
                    self.center is not None
                ):  # found bottom and center
                    _bottom_origin = self.set_point_2_origin(
                        origin=self.get_center_point(self.center),
                        point=self.get_center_point(self.bottom),
                    )

                    _head_origin = (-1 * _bottom_origin[0], -1 * _bottom_origin[1])
                    _head_point = self.set_origin_2_point(
                        origin=self.get_center_point(self.center), point=_head_origin
                    )
                    self.head = np.array(
                        [_head_point[0], _head_point[1], _head_point[0], _head_point[1]]
                    )

                elif (self.center is not None) and (
                    self.needle is not None
                ):  # found center and needle
                    _needle_origin = self.set_point_2_origin(
                        origin=self.get_center_point(self.center),
                        point=self.get_center_point(self.needle),
                    )
                    _head_origin = (_needle_origin[0], _needle_origin[1])
                    _head_point = self.set_origin_2_point(
                        origin=self.get_center_point(self.center), point=_head_origin
                    )
                    self.head = np.array(
                        [_head_point[0], _head_point[1], _head_point[0], _head_point[1]]
                    )

                    self.needle = None

            if self.min is None and (self.max is not None and self.center is not None):
                _max_origin = self.set_point_2_origin(
                    origin=self.get_center_point(self.center),
                    point=self.get_center_point(self.max),
                )
                _min_origin = (-1 * _max_origin[0], _max_origin[1])
                _min_point = self.set_origin_2_point(
                    origin=self.get_center_point(self.center), point=_min_origin
                )
                self.min = np.array(
                    [_min_point[0], _min_point[1], _min_point[0], _min_point[1]]
                )

            if self.max is None and (self.min is not None and self.center is not None):
                _min_origin = self.set_point_2_origin(
                    origin=self.get_center_point(self.center),
                    point=self.get_center_point(self.min),
                )
                _max_origin = (-1 * _min_origin[0], _min_origin[1])
                _max_point = self.set_origin_2_point(
                    origin=self.get_center_point(self.center), point=_max_origin
                )
                self.max = np.array(
                    [_max_point[0], _max_point[1], _max_point[0], _max_point[1]]
                )
            

    def preprocess_part(
        self,
    ):
        return
    
    def preprocess_circle(self,):
        ic(f"preprocess circle.")
        # don't need to use bottom

        # not found min -> min = max
        if (self.min is None) and (self.max is not None):
            self.min = self.max 
        
        # not found max -> max = min   
        if (self.max is None) and (self.min is not None):
            self.max = self.min
        
        if self.center is None:  # not found center
                if (self.bottom is not None) and (
                    self.head is not None
                ):  # found bottom and head
                    _bottom_point = self.get_center_point(self.bottom)
                    _head_point = self.get_center_point(self.head)

                    # make the average 2 round -> bottom is nearer center than head
                    _avg_bottom_head = self.get_center_point(
                        np.array(
                            [
                                _bottom_point[0],
                                _bottom_point[1],
                                _head_point[0],
                                _head_point[1],
                            ]
                        )
                    )
                    _avg_bottom_head = self.get_center_point(
                        np.array(
                            [
                                _bottom_point[0],
                                _bottom_point[1],
                                _avg_bottom_head[0],
                                _avg_bottom_head[1],
                            ]
                        )
                    )

                    self.center = np.array(
                        [
                            _avg_bottom_head[0],
                            _avg_bottom_head[1],
                            _avg_bottom_head[0],
                            _avg_bottom_head[1],
                        ]
                    )
                elif (self.head is not None) and (
                    self.needle is not None
                ):  # foud head and needle
                    # center is point(from needle) that farest from head
                    _center_head = self.get_center_point(point=self.head)
                    n_x_min, n_y_min, n_x_max, n_y_max = self.needle
                    # find 4 point of needle
                    n_tl = (n_x_min, n_y_min)  # top left point
                    n_tr = (n_x_max, n_y_min)  # top righ
                    n_bl = (n_x_min, n_y_max)  # bottom left point
                    n_br = (n_x_max, n_y_max)  # bottom righ point

                    needle_point = [n_tl, n_tr, n_bl, n_br]  # list of needle point
                    needle_point_dict = dict(
                        zip(
                            needle_point,
                            [
                                self.get_distance_from_2_point(
                                    point1=_center_head, point2=_p
                                )
                                for _p in needle_point
                            ],
                        )
                    )  # dict key=point , value=distance
                    needle_max_dis_point = max(
                        needle_point_dict, key=lambda k: needle_point_dict[k]
                    )

                    self.center = np.array(
                        [
                            needle_max_dis_point[0],
                            needle_max_dis_point[1],
                            needle_max_dis_point[0],
                            needle_max_dis_point[1],
                        ]
                    )

                    # ic(needle_point)
                    # ic(needle_point_dict)
                    # ic(needle_max_dis_point)
                    # max_dis_point = needle_point[0]
                    # max_dis=0
                    # for _needle_p in needle_point:
                    #     _dis = self.get_distance_from_2_point(point1=_center_head , point2=_needle_p)
                    #     if _dis > max_dis:

        if self.head is None:
            if (self.bottom is not None) and (
                self.center is not None
            ):  # found bottom and center
                _bottom_origin = self.set_point_2_origin(
                    origin=self.get_center_point(self.center),
                    point=self.get_center_point(self.bottom),
                )

                _head_origin = (-1 * _bottom_origin[0], -1 * _bottom_origin[1])
                _head_point = self.set_origin_2_point(
                    origin=self.get_center_point(self.center), point=_head_origin
                )
                self.head = np.array(
                    [_head_point[0], _head_point[1], _head_point[0], _head_point[1]]
                )

            elif (self.center is not None) and (
                self.needle is not None
            ):  # found center and needle
                _needle_origin = self.set_point_2_origin(
                    origin=self.get_center_point(self.center),
                    point=self.get_center_point(self.needle),
                )
                _head_origin = (_needle_origin[0], _needle_origin[1])
                _head_point = self.set_origin_2_point(
                    origin=self.get_center_point(self.center), point=_head_origin
                )
                self.head = np.array(
                    [_head_point[0], _head_point[1], _head_point[0], _head_point[1]]
                )

                self.needle = None

        
    def predict_clock(
        self,
    ):
        if self.case == "normal":
            return self.predict_clock_normal()
        if self.case == "circle":
            return self.predict_clock_circle()

        return 0

    def predict_clock_normal(
        self,
    ):

        if self.check_attribute_is_none():
            print(f"can not predict -> lack of class")
            return 0
        
        # check needle is lower than min
        max_min_angle, _ = self.angle_between_2_vector(
            start_vector=self.set_point_2_origin(
                    origin=self.get_center_point(self.center),
                    point=self.get_center_point(self.max),
                    
                ),
            end_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.min)
            )
        )

        max_head_angle, _ = self.angle_between_2_vector(
            start_vector=self.set_point_2_origin(
                    origin=self.get_center_point(self.center),
                    point=self.get_center_point(self.max),
                    
                ),
            end_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.head)
            )
        )

        head_max_angle, _ = self.angle_between_2_vector(
            start_vector=self.set_point_2_origin(
                    origin=self.get_center_point(self.center),
                    point=self.get_center_point(self.head),
                    
                ),
            end_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.max)
            )
        )


        if (max_head_angle >= max_min_angle) and (abs(max_head_angle - max_min_angle) <= head_max_angle):
            # print(f"--> needle is lower than min.")
            return self.min_value 
        
        # -------------------------------
        
        # check needle is higher than max

        head_min_angle, _ = self.angle_between_2_vector(
            start_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.head),
            ),
            end_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.min),
            ),
        )

        min_head_angle, _ = self.angle_between_2_vector(
            start_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.min),
            ),
            end_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.head),
            ),
        )


        if (head_min_angle >= max_min_angle) and (abs(head_min_angle - max_min_angle) <= min_head_angle ):
            # print(f"--> needle is higher than max")
            return self.max_value

        # -------------------------------

        head_min_angle, _ = self.angle_between_2_vector(
            start_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.head),
            ),
            end_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.min),
            ),
        )
        # ic(head_min_angle)

        max_head_angle, _ = self.angle_between_2_vector(
            start_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.max),
            ),
            end_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.head),
            ),
        )
        # ic(max_head_angle)

        all_angle = head_min_angle + max_head_angle
        clock_ratio = head_min_angle / all_angle

        if self.min_max_swap:
            clock_ratio = 1 - clock_ratio

        # ic(clock_ratio)

        range_value = self.max_value - self.min_value
        actual_value = clock_ratio * range_value + self.min_value
        # ic(actual_value)

        return actual_value
    
    
    def predict_clock_circle(
        self,
    ):
        if self.check_attribute_is_none():
            print(f"can not predict -> lack of class")
            return 0
        
        head_min_angle, _ = self.angle_between_2_vector(
            start_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.head),
            ),
            end_vector=self.set_point_2_origin(
                origin=self.get_center_point(self.center),
                point=self.get_center_point(self.min),
            ),
        )

        ic(f"head_min_angle: {head_min_angle}")

        clock_ratio = head_min_angle / 360.0

        range_value = self.max_value - self.min_value
        actual_value = clock_ratio * range_value + self.min_value

        return actual_value  

    def angle_between_2_vector(
        self, start_vector, end_vector, step_deg=0.5, min_deg_thres=0
    ):
        temp_vector = start_vector
        for deg in [x * step_deg for x in range(0, int(360 / step_deg) + 1)]:
            if (
                int(self.angle_between_2_line(a=temp_vector, b=end_vector))
                > min_deg_thres
            ):
                temp_vector = self.rotation_vector(
                    theta=np.deg2rad(deg), vector=start_vector
                )
            else:
                return deg, temp_vector

        return 0

    def rotation_vector(self, theta, vector):  # theta in radians
        rotation_matrix = np.array(
            [
                [math.cos(theta), -1 * math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ]
        )

        vector_use = np.array(vector)

        # print(f"rotation matrix shape: {rotation_matrix.shape}")
        # print(f"vector_use shape: {vector_use.shape}")

        rotated_vector = rotation_matrix @ vector_use

        return rotated_vector

    def angle_between_2_line(self, a, b):
        # convert a and b from tuple to numpy array
        vec_a = np.array(a)
        vec_b = np.array(b)

        dot_ab = vec_a @ vec_b
        magnitude_a = self.get_distance_from_2_point(point1=(0, 0), point2=a)
        magnitude_b = self.get_distance_from_2_point(point1=(0, 0), point2=b)

        rad = np.arccos(dot_ab / (magnitude_a * magnitude_b))
        degree = np.rad2deg(rad)

        return degree

    def get_distance_from_2_point(self, point1: tuple, point2: tuple):
        """
        point1: tuple (x,y)
        point2: tuple (x,y)
        """
        distance = math.sqrt(
            (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        )
        return distance

    def set_origin_2_point(self, origin, point):
        return ((point[0] + origin[0]), (-1 * point[1] + origin[1]))

    def set_point_2_origin(self, origin, point):
        return (point[0] - origin[0], -1 * (point[1] - origin[1]))

    def get_center_point(self, point):
        return (
            point[0] + (point[2] - point[0]) / 2,
            point[1] + (point[3] - point[1]) / 2,
        )

    def visualize_clock(self, image=None):
        # create mock image
        if image is None:
            image = np.ones((640, 640))

        plt.imshow(image)

        if self.min is not None:
            min_x, min_y = self.get_center_point(self.min)
            plt.plot(min_x, min_y, "ro", markersize=10)
            plt.annotate(
                "min",
                (min_x, min_y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        if self.max is not None:
            max_x, max_y = self.get_center_point(self.max)
            plt.plot(max_x, max_y, "ro", markersize=10)
            plt.annotate(
                "max",
                (max_x, max_y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        if self.center is not None:
            center_x, center_y = self.get_center_point(self.center)
            plt.plot(center_x, center_y, "ro", markersize=10)
            plt.annotate(
                "center",
                (center_x, center_y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        if self.head is not None:
            head_x, head_y = self.get_center_point(self.head)
            plt.plot(head_x, head_y, "ro", markersize=10)
            plt.annotate(
                "head",
                (head_x, head_y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        #TODO: you can comment this
        if self.bottom is not None:

            bottom_x, bottom_y = self.get_center_point(self.bottom)
            plt.plot(bottom_x, bottom_y, "ro", markersize=10)
            plt.annotate(
                "bottom",
                (bottom_x, bottom_y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        if self.needle is not None:
            needle_x, needle_y = self.get_center_point(self.needle)
            plt.plot(needle_x, needle_y, "ro", markersize=10)
            plt.annotate(
                "needle",
                (needle_x, needle_y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        # if self.center is not None and self.min is not None:
        #     plt.plot(
        #         (self.center[0], self.min[0]),
        #         (self.center[1], self.min[1]),
        #         linestyle="-",
        #         color="blue",
        #     )  # center min line

        # if self.center is not None and self.head is not None:
        #     plt.plot(
        #         (self.center[0], self.head[0]),
        #         (self.center[1], self.head[1]),
        #         linestyle="-",
        #         color="blue",
        #     )  # center head line

        # if self.center is not None and self.max is not None:
        #     plt.plot(
        #         (self.center[0], self.max[0]),
        #         (self.center[1], self.max[1]),
        #         linestyle="-",
        #         color="blue",
        #     )  # center max line

        plt.show()

    def print(
        self,
    ):
        print(f"gauge: {self.gauge}")
        print(f"min: {self.min}")
        print(f"max: {self.max}")
        print(f"center: {self.center}")
        print(f"head: {self.head}")
        print(f"bottom: {self.bottom}")
        print(f"needle: {self.needle}")

    def check_attribute_is_none(
        self,
    ):
        # if lack of class -> True
        if (
            (self.min is not None)
            and (self.max is not None)
            and (self.head is not None)
            and (self.center is not None)
        ):
            return False

        return True
    