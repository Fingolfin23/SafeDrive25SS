import obstacle_avoidance.geometric_primitives as gp
import obstacle_avoidance.obstacles as obstcl


def get_custom_interface_methods(obj: object):
    return list(filter(lambda x: not x.startswith('__'), dir(obj)))


def check_obstacles(obstacles: list):
    obstacle_methods = get_custom_interface_methods(obstcl.ObstacleInterface())
    for i, obstacle in enumerate(obstacles):
        for method in obstacle_methods:
            assert method in dir(obstacle), f'obstacle: {i}, does not implement method: {method}'


class DistanceDetector:
    def __init__(self, obstacles: list):
        check_obstacles(obstacles)
        self.obstacles = obstacles

    def get_min_distance_vector(self, v: gp.Vector):
        """
        Returns minimal distance to obstacles from vector v.
        If vector v does not reach any obstacle, vector v is returned.
        """

        result = v
        for obstacle in self.obstacles:
            cur_v = obstacle.get_intersection_with_vector(result)
            if cur_v is not None:
                result = cur_v
        return result


class FieldOfViewVectorSampler:
    _DISTRIBUTION_UNIFORM = 'uniform'
    _DISTRIBUTION_TYPES = [
        _DISTRIBUTION_UNIFORM
    ]

    def __init__(
        self,
        view_point: gp.Point,
        direction: gp.Vector,
        distance: float,
        view_angle: float,
    ):
        """
        Params:
          - view_point - coordinate of viewer
          - direction - direction of the view (
                central direction of the view, field of view boundaries are equally far from the central direction
                origin and length are ommited
            )
          - distance - max distance of the field of view
          - view_angle - angle of the view in radians (direction splits view_angle in two equal parts)
        """
        assert view_angle > gp.EPS, f'view_ange has to be positive float'

        self.view_point = view_point
        self.direction = direction.get_position()
        self.distance = distance
        self.view_angle = view_angle
    
    def sample_view_vectors(self, count: int, distribuiton: str = _DISTRIBUTION_UNIFORM):
        """
            Returns list of view vectors distributed inside field of view
            Params:
              - count - count of view vectors
              - distribution - enum (`uniform` - distributed with constant step)
        """
        assert count > 2
        assert distribuiton in FieldOfViewVectorSampler._DISTRIBUTION_TYPES, f'distribution: {distribuiton} is not valid distribution type'

        angles = None
        if distribuiton == FieldOfViewVectorSampler._DISTRIBUTION_UNIFORM:
            angle_step = self.view_angle / (count - 1)
            angles = [i * angle_step for i in range(count)]
        else:
            assert False, 'Unreachable distribution condition'

        init_vector = self.direction.rotate_by(-self.view_angle / 2.0)
        return [
            init_vector.rotate_by(angle).shift_to_new_origin(self.view_point).update_length(self.distance)
            for angle in angles
        ]
