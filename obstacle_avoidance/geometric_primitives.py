import math


EPS = 1e-6


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return math.fabs(self.x - other.x) < EPS and math.fabs(self.y - other.y) < EPS

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __repr__(self):
        return str(self)


class Vector:
    def __init__(self, x: float, y: float, origin: Point = Point(0, 0)):
        self.origin = origin
        self.x = x
        self.y = y
    
    def norm(self) -> float:
        """
        Returns vector norm
        """
        return math.sqrt((self.x - self.origin.x) ** 2 + (self.y - self.origin.y) ** 2)

    def get_position(self):
        """
        Returns vector from the origin
        """
        return Vector(self.x - self.origin.x, self.y - self.origin.y)

    def rotate_by(self, phi: float):
        """
        Rotates the vector by angle phi (in radians)
        """
        pos = self.get_position()
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)

        new_x = pos.x * cos_phi - pos.y * sin_phi
        new_y = pos.x * sin_phi + pos.y * cos_phi
        return Vector(new_x + self.origin.x, new_y + self.origin.y, self.origin)

    def shift_to_new_coordinates(self, x: float, y: float):
        """
        Method updates all data to shift vector from old origin to new_origin.
        """
        diff_x = x - self.x
        diff_y = y - self.y
        return self.shift_to_new_origin(Point(self.origin.x + diff_x, self.origin.y + diff_y))

    def shift_to_new_origin(self, new_origin: Point):
        """
        Method updates all data to shift vector from old origin to new_origin.
        """
        new_vector_point = Point(self.x, self.y) - self.origin + new_origin
        return Vector(new_vector_point.x, new_vector_point.y, new_origin)

    def shift_parallel(self, p: Point):
        """
        Method shifts the vector parallel.
        p - point of the parallel line, on which shifted vector has to lay.
        """
        # assert self.norm() > EPS, 'vector length has to be positive'

        norm = Vector(self.y - self.origin.y, self.origin.x - self.x)
        t = (norm.x * (p.x - self.origin.x) + norm.y * (p.y - self.origin.y)) / (norm.x ** 2 + norm.y ** 2)
        new_start = Vector(self.origin.x, self.origin.y) + norm * t
        new_end = Vector(self.x, self.y) + norm * t
        return Vector(new_end.x, new_end.y, Point(new_start.x, new_start.y))

    def update_length(self, new_length: float):
        """
        Method updates vector length (origin is stable)
        """
        assert new_length > EPS, f'vector length must be more than zero ({new_length} > 0 violated)'
        return self * (new_length / self.norm())

    def get_angle_between(self, other):
        other_pos = other.get_position()
        pos = self.get_position()
        return math.atan2(other_pos.y, other_pos.x) - math.atan2(pos.y, pos.x)

    def __mul__(self, resize_coef: float):
        """
        Multipies length of the vector by the resize_coef.
        """
        pos = self.get_position()
        return Vector(pos.x * resize_coef, pos.y * resize_coef).shift_to_new_origin(self.origin)

    def __neg__(self):
        return self * (-1)

    def __add__(self, other):
        pos = other.get_position()
        return Vector(self.x + pos.x, self.y + pos.y, self.origin)

    def __sub__(self, other):
        return self + (-other)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return Point(self.x, self.y) == Point(other.x, other.y) and self.origin == other.origin

    def __str__(self):
        return f'({self.x}, {self.y}), orig: {str(self.origin)}'

    def __repr__(self):
        return str(self)
