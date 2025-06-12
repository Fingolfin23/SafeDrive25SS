from . import geometric_primitives as gp

import math


class ObstacleInterface:
    def get_intersection_with_vector(
            self,
            v: gp.Vector,
            is_rectangularized: bool = False,
        ) -> gp.Vector:
        """
        Method returns point on vector v, where the vector v intersects with obstacle.
        If vector v does not intersect with the obstacle then None is returned.
        is_rectangularized if True then is returned not real intersection wtih the obstacle,
        but intersection with rectangular boundary comprising the obstacle.
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

    def _get_rectangularized_intersection(self, v, real_intersection):
        centerized_v = v.shift_parallel(self.center)
        intersection_vector = self._get_real_intersection(centerized_v)

        if intersection_vector is None:
            return None

        if real_intersection.norm() < intersection_vector.norm():
            return real_intersection

        return intersection_vector.shift_parallel(v.origin)

    def get_intersection_with_vector(
        self,
        v: gp.Vector,
        is_rectangularized: bool = False,
    ) -> gp.Vector:
        """
        Method returns point on vector v, where the vector v intersects with obstacle.
        If vector v does not intersect with the obstacle then None is returned.
        """

        assert v.norm() > gp.EPS, 'vector has to have positive length'

        intersection = self._get_real_intersection(v)
        if is_rectangularized and intersection is not None:
            return self._get_rectangularized_intersection(v, intersection)
        return intersection

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
