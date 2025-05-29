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

    def shift_to_new_origin(self, new_origin: Point):
        """
        Method updates all data to shift vector from old origin to new_origin.
        """
        new_vector_point = Point(self.x, self.y) - self.origin + new_origin
        return Vector(new_vector_point.x, new_vector_point.y, new_origin)

    def update_length(self, new_length: float):
        """
        Method updates vector length (origin is stable)
        """
        assert new_length > EPS, f'vector length must be more than zero ({new_length} > 0 violated)'

        pos = self.get_position()
        resize_coef = new_length / pos.norm()
        return Vector(pos.x * resize_coef, pos.y * resize_coef).shift_to_new_origin(self.origin)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return Point(self.x, self.y) == Point(other.x, other.y) and self.origin == other.origin

    def __str__(self):
        return f'({self.x}, {self.y}), orig: {str(self.origin)}'

    def __repr__(self):
        return str(self)
