import math
import numpy as np

from . import geometric_primitives as gp


class PathReader:
    def __init__(self, path_filename: str):
        data = np.loadtxt(path_filename, delimiter=',')

        self.x_coordinates = data[:, 0]
        self.y_coordinates = data[:, 1]
        self.dist_to_right_bound = data[:, 2]
        self.dist_to_left_bound = data[:, 3]

        self.right_bound_raw = PathReader._build_bound(
            self.x_coordinates,
            self.y_coordinates,
            self.dist_to_right_bound,
            -math.pi/2,
        )
        self.left_bound_raw = PathReader._build_bound(
            self.x_coordinates,
            self.y_coordinates,
            self.dist_to_left_bound,
            math.pi/2,
        )

    @staticmethod
    def _build_bound(
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
        dist_to_bound: np.ndarray,
        alpha: float,
    ):
        result = []
        for i in range(1, x_coordinates.shape[0]):
            v = gp.Vector(
                x_coordinates[i],
                y_coordinates[i],
                gp.Point(x_coordinates[i - 1], y_coordinates[i - 1]),
            )
            v = v.rotate_by(alpha).update_length(dist_to_bound[i - 1])
            result.append([v.x, v.y])
        v = gp.Vector(
            x_coordinates[-1],
            y_coordinates[-1],
            gp.Point(x_coordinates[-2], y_coordinates[-2]),
        )
        v = v.shift_to_new_origin(gp.Point(x_coordinates[-1], y_coordinates[-1]))
        v = v.rotate_by(alpha).update_length(dist_to_bound[-1])
        result.append([v.x, v.y])
        return np.asarray(result)
