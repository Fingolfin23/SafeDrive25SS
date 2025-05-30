import obstacle_avoidance.geometric_primitives as gp
import obstacle_avoidance.field_of_view_analyzer as fva

import math
import numpy as np


class ObstacleAwareTrackBuilder:
    def __init__(self, obstacles: list):
        self.distance_detector = fva.DistanceDetector(obstacles)

    @staticmethod
    def _get_step_vector(
        prev_central_path_point,
        cur_central_path_point,
        extra_points_number,
    ):
        """
        Returns step vector between adjacent points of the central path.
        """
        return gp.Vector(
            cur_central_path_point[0],
            cur_central_path_point[1],
            gp.Point(prev_central_path_point[0], prev_central_path_point[1]),
        ) * (1.0 / (extra_points_number + 1))

    @staticmethod
    def _get_full_width_normal_vector(
        cur_step_vector,
        distance_to_path_bound,
    ):
        """
        Returns the vector, which is perpendicular to the direction
          of the central path in the point of the current step vector (cur_step_vector.x, cur_step_vector.y).
        Resulting vector has length of 2 * distance_to_path_bound,
          starts at the left bound of the track and ends at the right bound.
        """
        normal_vector = cur_step_vector.shift_to_new_origin(
            gp.Point(cur_step_vector.x, cur_step_vector.y)
        ).rotate_by(
            math.pi / 2
        ).update_length(distance_to_path_bound)
        return gp.Vector(
            cur_step_vector.x,
            cur_step_vector.y,
            gp.Point(normal_vector.x, normal_vector.y),
        ).update_length(distance_to_path_bound * 2)

    @staticmethod
    def _find_center_point_of_available_gap(
        distance_detector,
        normal_vector,
        minimal_gap_width,
        is_rectagularized,
    ):
        obstacles_set = set()
        max_gap_vector = gp.Vector(0, 0)
        while True:
            cur_gap_vector, obstacle_index = distance_detector.get_min_distance_vector(
                normal_vector,
                is_rectagularized=is_rectagularized,
            )
            # if start of the vector is outside of any obstacle and width of the current gap is greater than previously found gap
            if len(obstacles_set) == 0 and max_gap_vector.norm() < cur_gap_vector.norm():
                max_gap_vector = cur_gap_vector

            if obstacle_index is None:  # no obstacles occurred after previous step
                break

            if obstacle_index in obstacles_set:
                obstacles_set.remove(obstacle_index)
            else:
                obstacles_set.add(obstacle_index)
            normal_vector.origin = gp.Point(cur_gap_vector.x, cur_gap_vector.y)
            normal_vector = (normal_vector * (1 - gp.EPS)).shift_to_new_coordinates(
                normal_vector.x,
                normal_vector.y,
            )

        if minimal_gap_width > max_gap_vector.norm() + gp.EPS:  # path does not meet required minimal_gap_width condition
            return [None, None], 0

        gap_center_vector = max_gap_vector * 0.5
        return [gap_center_vector.x, gap_center_vector.y], gap_center_vector.norm()

    def rebuild_central_path(
        self,
        central_path: np.ndarray,
        distance_to_path_bound: float,
        minimal_gap_width: float,
        extra_points_number: int = 0,
        is_rectagularized: bool = False,
    ):
        """
            Rebuilds central path to avoid all obstacles along the whole path.
            Params:
              - central_path - array of shape (N, 2) containing coordinates of points along central path
              - distance_to_path_bound - distance to bound from the central path
              - minimal_gap_width - minimal gap width required for avoiding obstacles
              - extra_points_number - how much points are added between initial points (from central_path)
        """
        if not isinstance(central_path, np.ndarray):
            central_path = np.asarray(central_path)

        assert central_path.shape[0] > 2 and central_path.shape[1] == 2, f'central_path shape has to be (N, 2), N > 2 (value {central_path.shape} is invalid)'
        assert distance_to_path_bound > gp.EPS, f'distance_to_path_bound has to be positive float (value {distance_to_path_bound} is invalid)'
        assert minimal_gap_width > gp.EPS, f'minimal_gap_width has to be positive float (value {minimal_gap_width} is invalid)'
        assert extra_points_number >= 0, f'extra_points_number has to be >= 0, (value {extra_points_number} is invalid)'

        result = [central_path[0]]  # starting point does not change
        if minimal_gap_width > distance_to_path_bound * 2 + gp.EPS:
            result = [[None, None]]
        new_distance_to_path_bound = distance_to_path_bound
        for i in range(1, central_path.shape[0]):
            step_vector = ObstacleAwareTrackBuilder._get_step_vector(
                prev_central_path_point=central_path[i - 1],
                cur_central_path_point=central_path[i],
                extra_points_number=extra_points_number,
            )
            for k in range(1, extra_points_number + 2):
                cur_step_vector = step_vector * k
                normal_vector = ObstacleAwareTrackBuilder._get_full_width_normal_vector(
                    cur_step_vector=cur_step_vector,
                    distance_to_path_bound=distance_to_path_bound,
                )
                center_point, cur_distance_to_path_bound = ObstacleAwareTrackBuilder._find_center_point_of_available_gap(
                    self.distance_detector,
                    normal_vector,
                    minimal_gap_width,
                    is_rectagularized,
                )
                new_distance_to_path_bound = min(cur_distance_to_path_bound, new_distance_to_path_bound)
                result.append(center_point)
        return np.asarray(result, dtype=np.float32), new_distance_to_path_bound
