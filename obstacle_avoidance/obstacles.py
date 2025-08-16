from . import geometric_primitives as gp
from . import path_reader

import math
import numpy as np


class ObstacleInterface:
    def get_intersection_with_vector(
            self,
            v: gp.Vector,
    ) -> gp.Vector:
        """
        Method returns point on vector v, where the vector v intersects with obstacle.
        If vector v does not intersect with the obstacle then None is returned.
        """
        pass

    def get_tangent_vectors(
            self,
            p: gp.Point,
            length: float,
    ) -> tuple[gp.Vector, gp.Vector]:
        """
        Method returns tangent vectors to obstacle of length size (from point p).
        """
        pass

    def shift_to_new_origin(self, new_origin: gp.Point):
        """
        Method updates all data to shift obstacle from old origin (0, 0) to new_origin.
        """
        pass

    def is_inside(self, p: gp.Point):
        """
        Method returns True, if point p is inside the obstacle.
        """
        pass


class CircleObstacle(ObstacleInterface):
    def __init__(self, center: gp.Point, radius: float):
        assert radius > -gp.EPS, 'radius has to be greater than zero'
        
        self.center = center
        self.radius = radius

    def _get_real_intersection(self, v):
        v_pos = v.get_position()
        circle_pos = self.shift_to_new_origin(v.origin)
        center = circle_pos.center
        v_pos_norm = v_pos.norm()
        
        distance = math.fabs(-v_pos.y * center.x + v_pos.x * center.y) / v_pos_norm

        if distance > circle_pos.radius + gp.EPS:  # distance to vector line is more than radius
            return None

        # calculation intersection points
        v_pos_norm_sq = v_pos_norm ** 2
        discriminant = math.sqrt(
            v_pos_norm_sq * circle_pos.radius ** 2 - (v_pos.y * center.x - v_pos.x * center.y) ** 2
        )
        scalar_mul = v_pos.x * center.x + v_pos.y * center.y

        t1 = (scalar_mul + discriminant) / v_pos_norm_sq
        t2 = (scalar_mul - discriminant) / v_pos_norm_sq

        if t1 < -gp.EPS and t2 < -gp.EPS:  # both intersection points lay outside of vector direction
            return None
        
        t = min(t1, t2)
        if t < -gp.EPS:  # if one of intersection points lays outside of vector direction
            t = t1 + t2 - t  # change to different t_i value

        dist_vector = gp.Vector(v_pos.x * t, v_pos.y * t)
        if dist_vector.norm() > v_pos_norm + gp.EPS:  # initial vector is shorter than found vector
            return None

        return dist_vector.shift_to_new_origin(v.origin)

    def get_intersection_with_vector(
        self,
        v: gp.Vector,
    ) -> gp.Vector:
        """
        Method returns point on vector v, where the vector v intersects with obstacle.
        If vector v does not intersect with the obstacle then None is returned.
        """

        assert v.norm() > gp.EPS, 'vector has to have positive length'

        return self._get_real_intersection(v)

    def get_tangent_vectors(
            self,
            p: gp.Point,
            length: float,
    ) -> tuple[gp.Vector, gp.Vector]:
        """
        Method returns tangent vectors to circle of length size (from point p).
        """
        assert length > gp.EPS, 'length has to be greater than zero'
        assert not self.is_inside(p), 'point has to be outside of circle'

        v = gp.Vector(self.center.x, self.center.y, p)
        alpha = math.asin(self.radius / v.norm())
        v = v.update_length(length)
        return v.rotate_by(alpha + gp.EPS), v.rotate_by(-alpha - gp.EPS)

    def is_inside(self, p: gp.Point):
        """
        Method returns True, if point p is inside the obstacle.
        """
        return (p.x - self.center.x) ** 2 + (p.y - self.center.y) ** 2 < self.radius ** 2

    def shift_to_new_origin(self, new_origin: gp.Point):
        """
        Method updates all data to shift obstacle from old origin (0, 0) to new_origin.
        """
        return CircleObstacle(self.center - new_origin, self.radius)

    def update_radius(self, new_radius: float):
        """
        Method updates radius of the circle obstacle.
        """
        return CircleObstacle(self.center, new_radius)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.center == other.center and math.fabs(self.radius - other.radius) < gp.EPS

    def __str__(self):
        return f'center: {str(self.center)}, radius: {self.radius}'

    def __repr__(self):
        return str(self)


class SegmentObstacle(ObstacleInterface):
    def __init__(self, p1: gp.Point, p2: gp.Point):
        self.p1 = p1
        self.p2 = p2

    def get_intersection_with_vector(
        self,
        v: gp.Vector,
    ) -> gp.Vector:
        """
        Method returns point on vector v, where the vector v intersects with obstacle.
        If vector v does not intersect with the obstacle then None is returned.
        """
        assert v.norm() > gp.EPS, 'vector has to have positive length'

        denominator = (self.p2.x - self.p1.x) * (v.y - v.origin.y) - (v.x - v.origin.x) * (self.p2.y - self.p1.y)
        if math.fabs(denominator) < gp.EPS:
            return None

        numerator = (v.origin.x - self.p1.x) * (self.p2.y - self.p1.y) - (self.p2.x - self.p1.x) * (v.origin.y - self.p1.y)
        t = numerator / denominator
        if t < -gp.EPS or t > 1 + gp.EPS:
            return None

        min_x = min(self.p1.x, self.p2.x)
        max_x = max(self.p1.x, self.p2.x)
        min_y = min(self.p1.y, self.p2.y)
        max_y = max(self.p1.y, self.p2.y)
        w = v * t
        if (min_x - gp.EPS < w.x and w.x < max_x + gp.EPS) and (min_y - gp.EPS < w.y and w.y < max_y + gp.EPS):
            return w

        return None

    def get_tangent_vectors(
            self,
            p: gp.Point,
            length: float,
    ) -> tuple[gp.Vector, gp.Vector]:
        """
        Method returns tangent vectors to segment of length size (from point p).
        """
        assert length > gp.EPS, 'length has to be greater than zero'
        assert not self.is_inside(p), 'point has to be outside of segment'

        v_left = gp.Vector(self.p1.x, self.p1.y, p)
        v_right = gp.Vector(self.p2.x, self.p2.y, p)
        
        v_left_pos = v_left.get_position()
        v_right_pos = v_right.get_position()
        v_left_atan = (math.atan2(v_left_pos.y, v_left_pos.x) + math.pi * 2) % (math.pi * 2)
        v_right_atan = (math.atan2(v_right_pos.y, v_right_pos.x) + math.pi * 2) % (math.pi * 2)
        if v_left_atan < v_right_atan:
            v_left, v_right = v_right, v_left

        return v_right.update_length(length).rotate_by(-gp.EPS), v_left.update_length(length).rotate_by(gp.EPS)

    def is_inside(self, p: gp.Point):
        """
        Method returns True, if point p is inside the obstacle.
        """
        return False

    def shift_to_new_origin(self, new_origin: gp.Point):
        """
        Method updates all data to shift obstacle from old origin (0, 0) to new_origin.
        """
        return SegmentObstacle(self.p1 - new_origin, self.p2 - new_origin)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.p1 == other.p1 and self.p2 == other.p2

    def __str__(self):
        return f'p1: {str(self.p1)}, p2: {str(self.p2)}'

    def __repr__(self):
        return str(self)


class TrackBoundObstacle(ObstacleInterface):
    def __init__(self, pr: path_reader.PathReader):
        if pr is None:
            self.right_bound_segments = []
            self.left_bound_segments = []
            return

        self.right_bound_segments = TrackBoundObstacle._build_segment_bound(
            pr.right_bound_raw
        )
        self.left_bound_segments = TrackBoundObstacle._build_segment_bound(
            pr.left_bound_raw
        )

    @staticmethod
    def _build_segment_bound(bound_raw: np.ndarray):
        result = []
        for i in range(1, bound_raw.shape[0]):
            result.append(
                SegmentObstacle(
                    gp.Point(bound_raw[i - 1][0], bound_raw[i - 1][1]),
                    gp.Point(bound_raw[i][0], bound_raw[i][1]),
                )
            )
        return result

    def get_intersection_with_vector(
        self,
        v: gp.Vector,
    ) -> gp.Vector:
        """
        Method returns point on vector v, where the vector v intersects with obstacle.
        If vector v does not intersect with the obstacle then None is returned.
        """
        assert v.norm() > gp.EPS, 'vector has to have positive length'

        def _iterate_over_seg_list(v, seg_list):
            for cur_segment in seg_list:
                res = cur_segment.get_intersection_with_vector(v)
                if res is not None:
                    v = res
            return v

        v = _iterate_over_seg_list(v, self.right_bound_segments)
        v = _iterate_over_seg_list(v, self.left_bound_segments)
        return v

    def get_tangent_vectors(
            self,
            p: gp.Point,
            length: float,
    ) -> tuple[gp.Vector, gp.Vector]:
        """
        Method returns tangent vectors to segment of length size (from point p).
        """
        pass

    def is_inside(self, p: gp.Point):
        """
        Method returns True, if point p is inside the obstacle.
        """
        pass

    def shift_to_new_origin(self, new_origin: gp.Point):
        """
        Method updates all data to shift obstacle from old origin (0, 0) to new_origin.
        """
        pass
