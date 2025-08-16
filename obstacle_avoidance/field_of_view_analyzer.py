from . import geometric_primitives as gp
from . import obstacles as obstcl
from . import path_reader

import copy
import collections
import math


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
        self.obstacles = copy.deepcopy(obstacles)
        self.safe_distance_from_obstacle = 0

    def set_safe_distance_from_obstacle(self, safe_distance_from_obstacle: float):
        assert safe_distance_from_obstacle > -gp.EPS, f'safe_distance_from_obstacle has to be >= 0, (value {safe_distance_from_obstacle} is invalid)'
        self.safe_distance_from_obstacle = safe_distance_from_obstacle
        for obstacle in self.obstacles:
            obstacle.radius += self.safe_distance_from_obstacle

    def unset_safe_distance_from_obstacle(self):
        for obstacle in self.obstacles:
            obstacle.radius -= self.safe_distance_from_obstacle

    def is_inside_obstacle(self, p: gp.Point):
        for obstacle in self.obstacles:
            if obstacle.is_inside(p):
                return True
        return False

    def get_min_distance_vector(
        self, v: gp.Vector,
    ) -> tuple[gp.Vector, int]:
        """
        Returns minimal distance to obstacles from vector v.
        If vector v does not reach any obstacle, vector v is returned.
        """

        result = v
        obstacle_index = None
        for index, obstacle in enumerate(self.obstacles):
            if result.norm() < gp.EPS:
                return result, obstacle_index

            cur_v = obstacle.get_intersection_with_vector(
                result,
            )
            if cur_v is not None:
                result = cur_v
                obstacle_index = index
        return result, obstacle_index


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
        assert view_angle > gp.EPS, 'view_ange has to be positive float'
        assert distance > gp.EPS, 'distance has to be positive float'

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
        assert count > 2, 'vectors count has to be > 2'
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


class FieldOfViewTangentVectorsExtractor:
    def __init__(self, obstacles: list):
        check_obstacles(obstacles)
        self.obstacles = copy.deepcopy(obstacles)
        self.safe_distance_from_obstacle = 0

    def set_safe_distance_from_obstacle(self, safe_distance_from_obstacle: float):
        assert safe_distance_from_obstacle > -gp.EPS, f'safe_distance_from_obstacle has to be >= 0, (value {safe_distance_from_obstacle} is invalid)'
        self.safe_distance_from_obstacle = safe_distance_from_obstacle
        for obstacle in self.obstacles:
            obstacle.radius += self.safe_distance_from_obstacle

    def unset_safe_distance_from_obstacle(self):
        for obstacle in self.obstacles:
            obstacle.radius -= self.safe_distance_from_obstacle

    def is_inside_obstacle(self, p: gp.Point):
        for obstacle in self.obstacles:
            if obstacle.is_inside(p):
                return True
        return False

    def extract_tangtent_vectors_list(
        self,
        view_point: gp.Point,
        distance: float,
    ):
        """
        Params:
          - view_point - coordinate of viewer
          - distance - max distance of the field of view
        """
        assert distance > gp.EPS, 'distance has to be positive float'

        tangent_vectors = []
        for i, obstacle in enumerate(self.obstacles):
            if self.is_inside_obstacle(view_point):
                continue

            for t in obstacle.get_tangent_vectors(view_point, distance):
                tangent_vectors.append((t, i))

        return tangent_vectors


class GapParams:
    def __init__(
        self,
        left_bound: gp.Vector,
        right_bound: gp.Vector,
    ):
        assert left_bound.origin == right_bound.origin, 'left_bound.origin and right_bound.origin have to be the same point'
        self.left_bound = left_bound
        self.right_bound = right_bound
        self._buffer = collections.deque([self.right_bound, self.left_bound])

    def get_gap_width(self):
        a = self.left_bound.norm()
        b = self.right_bound.norm()
        alpha = self.left_bound.get_angle_between(self.right_bound)
        return math.sqrt(a * a + b * b - 2 * a * b * math.cos(alpha))

    def shrink_extra_vector_length(self):
        """
        Updates lengths of vectors to the same value.
        """
        length = self.get_min_vector_length()
        return GapParams(
            left_bound=self.left_bound.update_length(length),
            right_bound=self.right_bound.update_length(length),
        )

    def flush_buffer(self):
        return GapParams(
            left_bound=self.left_bound,
            right_bound=self.right_bound,
        )

    def get_min_vector_length(self):
        return min(self.left_bound.norm(), self.right_bound.norm())

    def is_shrinkable(self):
        return len(self._buffer) > 2

    def append(self, v):
        if v == self.left_bound:
            return

        self._buffer.append(v)
        self.left_bound = self._buffer[-1]

    def pop(self):
        assert len(self._buffer) > 2, f'gap_params buffer has only {len(self._buffer)} items'
        self._buffer.pop()
        self.left_bound = self._buffer[-1]

    def popleft(self):
        assert len(self._buffer) > 2, f'gap_params buffer has only {len(self._buffer)} items'
        self._buffer.popleft()
        self.right_bound = self._buffer[0]

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.left_bound == other.left_bound and self.right_bound == other.right_bound

    def __str__(self):
        return f'left_bound: {str(self.left_bound)}, right_bound: {self.right_bound}'

    def __repr__(self):
        return str(self)


class FieldOfViewAnalyzer:
    _FOV_METHOD_VECTOR_SAMPLER = 'vector_sampler'
    _FOV_METHOD_TANGENT_VECTORS = 'tangent_vectors'
    _FOV_METHODS = [
        _FOV_METHOD_VECTOR_SAMPLER,
        _FOV_METHOD_TANGENT_VECTORS,
    ]

    def __init__(
        self,
        pr: path_reader.PathReader,
        obstacles: list,
        fov_method: str = _FOV_METHOD_VECTOR_SAMPLER,
        fov_sampler_distribution: str = FieldOfViewVectorSampler._DISTRIBUTION_UNIFORM,
        fov_sampler_count: int = 2,
        safe_distance_from_obstacle: float = 0,
    ):
        """
        Params:
          - pr - object of PathReader type to holding path params
          - obstacles - list of obstacles of type obstcl.ObstacleInterface
          - fov_method - type of method for FOV analysis
          - fov_sampler_distribution - type of distribution when _FOV_METHOD_VECTOR_SAMPLER is used
          - fov_sampler_count - number of sampled vectors when _FOV_METHOD_VECTOR_SAMPLER is used
        """
        assert fov_method in FieldOfViewAnalyzer._FOV_METHODS, f'fov_method = `{fov_method}` is incorrect'
        assert fov_sampler_distribution in FieldOfViewVectorSampler._DISTRIBUTION_TYPES, f'distribution: {fov_sampler_distribution} is not valid distribution type'
        assert fov_sampler_count > 2, 'vectors count has to be > 2'
        assert safe_distance_from_obstacle > -gp.EPS, 'safe_distance_from_obstacle has to be >= 0'

        # common params
        self.fov_method = fov_method
        self.safe_distance_from_obstacle = safe_distance_from_obstacle

        self.track_obstacle = obstcl.TrackBoundObstacle(pr)
        self.track_distance_detector = DistanceDetector([self.track_obstacle])
        self.obstacles = obstacles
        self.obstacles_distance_detector = DistanceDetector(self.obstacles)
        self.obstacles_distance_detector.set_safe_distance_from_obstacle(
            self.safe_distance_from_obstacle
        )

        # FieldOfViewVectorSampler params
        self.fov_sampler_distribution = fov_sampler_distribution
        self.fov_sampler_count = fov_sampler_count

        # FieldOfViewTangentVectorsExtractor params
        # ---

    def get_available_gap(
        self,
        view_point: gp.Point,
        direction: gp.Vector,
        distance: float,
        view_angle: float,
        minimal_gap_width: float,
    ) -> GapParams:
        """
          - view_point - coordinate of viewer
          - direction - direction of the view (
                central direction of the view, field of view boundaries are equally far from the central direction
                origin and length are ommited
            )
          - distance - max distance of the field of view
          - view_angle - angle of the view in radians (direction splits view_angle in two equal parts)
          - minimal_gap_width - minimal width, required for choosing a gap
        """
        assert view_angle > gp.EPS, 'view_ange has to be positive float'
        assert distance > gp.EPS, 'distance has to be positive float'
        assert minimal_gap_width > gp.EPS, 'minimal_gap_width has to be positive float'

        if self.fov_method == FieldOfViewAnalyzer._FOV_METHOD_VECTOR_SAMPLER:
            return self._get_gap_with_vector_sampler(
                view_point=view_point,
                direction=direction,
                distance=distance,
                view_angle=view_angle,
                minimal_gap_width=minimal_gap_width,
            )
        elif self.fov_method == FieldOfViewAnalyzer._FOV_METHOD_TANGENT_VECTORS:
            return self._get_gap_with_tangent_vectors(
                view_point=view_point,
                direction=direction,
                distance=distance,
                view_angle=view_angle,
                minimal_gap_width=minimal_gap_width,
            )
        else:
            raise RuntimeError(f'Unhandled fov_method: {self.fov_method}')

    def _detect_distance_common(self, vectors, minimal_gap_width):
        best_gap = None
        cur_gap = None

        def _update_best_gap():
            nonlocal cur_gap
            nonlocal best_gap

            if cur_gap is None:
                return

            if cur_gap.shrink_extra_vector_length().get_gap_width() + gp.EPS < minimal_gap_width:
                cur_gap = None
                return

            left_idx, right_idx = None, None
            max_len = 0
            for i, x in enumerate(cur_gap._buffer):
                if x.norm() > max_len:
                    left_idx = i
                    right_idx = i
                    max_len = x.norm()

            buf = cur_gap._buffer
            last_gap = GapParams(buf[left_idx], buf[right_idx])
            while 0 <= right_idx and left_idx < len(buf):
                right_v = buf[right_idx - 1] if right_idx > 0 else gp.Vector(0, 0)
                left_v = buf[left_idx + 1] if left_idx + 1 < len(buf) else gp.Vector(0, 0)
                if last_gap.get_gap_width() < minimal_gap_width + gp.EPS:
                    if right_v.norm() > left_v.norm():
                        last_gap = GapParams(last_gap.left_bound, right_v)
                        right_idx -= 1
                    else:
                        last_gap = GapParams(left_v, last_gap.right_bound)
                        left_idx += 1
                    continue

                if max(right_v.norm(), left_v.norm()) + gp.EPS < last_gap.get_min_vector_length():
                    break
            
                if right_v.norm() > left_v.norm():
                    last_gap = GapParams(last_gap.left_bound, right_v)
                    right_idx -= 1
                else:
                    last_gap = GapParams(left_v, last_gap.right_bound)
                    left_idx += 1

            cur_gap = last_gap.shrink_extra_vector_length()
            if best_gap is None:
                best_gap = cur_gap
                cur_gap = None
                return

            best_dist = best_gap.get_min_vector_length()
            cur_dist = cur_gap.get_min_vector_length()
            if best_dist + gp.EPS < cur_dist:
                best_gap = cur_gap
                cur_gap = None
                return

            if math.fabs(best_dist - cur_dist) < gp.EPS and best_gap.get_gap_width() + gp.EPS < cur_gap.get_gap_width():
                best_gap = cur_gap
                cur_gap = None
                return

            cur_gap = None

        for v in vectors:
            w, _ = self.obstacles_distance_detector.get_min_distance_vector(v)
            if v != w:
                # direction meets an obstacle
                _update_best_gap()
                continue

            w, _ = self.track_distance_detector.get_min_distance_vector(w)
            if cur_gap is None:
                cur_gap = GapParams(left_bound=w, right_bound=w)
            cur_gap.append(w)

        _update_best_gap()
        return best_gap

    def _get_gap_with_vector_sampler(
        self,
        view_point,
        direction,
        distance,
        view_angle,
        minimal_gap_width,
    ):
        fov_vector_sampler = FieldOfViewVectorSampler(
            view_point=view_point,
            direction=direction,
            distance=distance,
            view_angle=view_angle,
        )

        vectors = fov_vector_sampler.sample_view_vectors(
            count=self.fov_sampler_count,
            distribuiton=self.fov_sampler_distribution,
        )

        return self._detect_distance_common(vectors, minimal_gap_width)

    def _get_gap_with_tangent_vectors(
        self,
        view_point,
        direction,
        distance,
        view_angle,
        minimal_gap_width,
    ):
        # get view boundaries
        right_vector = direction.rotate_by(-view_angle / 2.0).shift_to_new_origin(view_point).update_length(distance)
        right_vector_pos = right_vector.get_position()
        right_atan = math.atan2(right_vector_pos.y, right_vector_pos.x)
        if right_atan < 0:
            right_atan += math.pi * 2

        left_vector = direction.rotate_by(view_angle / 2.0).shift_to_new_origin(view_point).update_length(distance)
        left_vector_pos = left_vector.get_position().rotate_by(-right_atan)
        left_atan = math.atan2(left_vector_pos.y, left_vector_pos.x)
        while left_atan < right_atan:
            left_atan += math.pi * 2
        right_atan = right_atan % (math.pi * 2)
        left_atan = left_atan % (math.pi * 2)

        # get tangent vectors
        vectors_with_atan = []
        for i, x in enumerate(
            FieldOfViewTangentVectorsExtractor(
                self.obstacles + self.track_obstacle.left_bound_segments + self.track_obstacle.right_bound_segments
            ).extract_tangtent_vectors_list(view_point, distance)
        ):
            cur_vector = x[0].update_length(distance)
            cur_vector_pos = cur_vector.get_position()
            cur_vector_atan = (math.atan2(cur_vector_pos.y, cur_vector_pos.x) + math.pi * 2) % (math.pi * 2)
            if right_atan <= cur_vector_atan and cur_vector_atan <= left_atan:
                vectors_with_atan.append((cur_vector, cur_vector_atan, i % 2 == 1, x[1]))

        vectors_with_atan.sort(key=lambda x: x[1])
        vectors = [right_vector] + list(map(lambda x: x[0], vectors_with_atan)) + [left_vector]
        return self._detect_distance_common(vectors, minimal_gap_width)
