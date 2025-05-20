import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import pathlib
import math
import bisect
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from enum import Enum, auto
from scipy.interpolate import CubicSpline
import casadi as ca
import matplotlib.animation as animation
import matplotlib.patches as patches
#======================
# author: Ke Xin
# Date: 20.05.2025
#======================
class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt


class QuarticPolynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time**2, 4 * time**3], [6 * time, 12 * time**2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class CartesianFrenetConverter:
    """
    A class for converting states between Cartesian and Frenet coordinate systems
    """

    @ staticmethod
    def cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa):
        """
        Convert state from Cartesian coordinate to Frenet coordinate

        Parameters
        ----------
            rs: reference line s-coordinate
            rx, ry: reference point coordinates
            rtheta: reference point heading
            rkappa: reference point curvature
            rdkappa: reference point curvature rate
            x, y: current position
            v: velocity
            a: acceleration
            theta: heading angle
            kappa: curvature

        Returns
        -------
            s_condition: [s(t), s'(t), s''(t)]
            d_condition: [d(s), d'(s), d''(s)]
        """
        dx = x - rx
        dy = y - ry

        cos_theta_r = math.cos(rtheta)
        sin_theta_r = math.sin(rtheta)

        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        d = math.copysign(math.hypot(dx, dy), cross_rd_nd)

        delta_theta = theta - rtheta
        tan_delta_theta = math.tan(delta_theta)
        cos_delta_theta = math.cos(delta_theta)

        one_minus_kappa_r_d = 1 - rkappa * d
        d_dot = one_minus_kappa_r_d * tan_delta_theta

        kappa_r_d_prime = rdkappa * d + rkappa * d_dot

        d_ddot = (-kappa_r_d_prime * tan_delta_theta +
                  one_minus_kappa_r_d / (cos_delta_theta * cos_delta_theta) *
                  (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa))

        s = rs
        s_dot = v * cos_delta_theta / one_minus_kappa_r_d

        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
        s_ddot = (a * cos_delta_theta -
                  s_dot * s_dot *
                  (d_dot * delta_theta_prime - kappa_r_d_prime)) / one_minus_kappa_r_d

        return [s, s_dot, s_ddot], [d, d_dot, d_ddot]

    @ staticmethod
    def frenet_to_cartesian(rs, rx, ry, rtheta, rkappa, rdkappa, s_condition, d_condition):
        """
        Convert state from Frenet coordinate to Cartesian coordinate

        Parameters
        ----------
            rs: reference line s-coordinate
            rx, ry: reference point coordinates
            rtheta: reference point heading
            rkappa: reference point curvature
            rdkappa: reference point curvature rate
            s_condition: [s(t), s'(t), s''(t)]
            d_condition: [d(s), d'(s), d''(s)]

        Returns
        -------
            x, y: position
            theta: heading angle
            kappa: curvature
            v: velocity
            a: acceleration
        """
        if abs(rs - s_condition[0]) >= 1.0e-6:
            raise ValueError(
                "The reference point s and s_condition[0] don't match")

        cos_theta_r = math.cos(rtheta)
        sin_theta_r = math.sin(rtheta)

        x = rx - sin_theta_r * d_condition[0]
        y = ry + cos_theta_r * d_condition[0]

        one_minus_kappa_r_d = 1 - rkappa * d_condition[0]

        tan_delta_theta = d_condition[1] / one_minus_kappa_r_d
        delta_theta = math.atan2(d_condition[1], one_minus_kappa_r_d)
        cos_delta_theta = math.cos(delta_theta)

        theta = CartesianFrenetConverter.normalize_angle(delta_theta + rtheta)

        kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]

        kappa = (((d_condition[2] + kappa_r_d_prime * tan_delta_theta) *
                  cos_delta_theta * cos_delta_theta) / one_minus_kappa_r_d + rkappa) * \
            cos_delta_theta / one_minus_kappa_r_d

        d_dot = d_condition[1] * s_condition[1]
        v = math.sqrt(one_minus_kappa_r_d * one_minus_kappa_r_d *
                      s_condition[1] * s_condition[1] + d_dot * d_dot)

        delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa

        a = (s_condition[2] * one_minus_kappa_r_d / cos_delta_theta +
             s_condition[1] * s_condition[1] / cos_delta_theta *
             (d_condition[1] * delta_theta_prime - kappa_r_d_prime))

        return x, y, theta, kappa, v, a

    @ staticmethod
    def normalize_angle(angle):
        """
        Normalize angle to [-pi, pi]
        """
        a = math.fmod(angle + math.pi, 2.0 * math.pi)
        if a < 0.0:
            a += 2.0 * math.pi
        return a - math.pi

class LateralMovement(Enum):
    HIGH_SPEED = auto()
    LOW_SPEED = auto()


class LongitudinalMovement(Enum):
    MERGING_AND_STOPPING = auto()
    VELOCITY_KEEPING = auto()


class CubicSpline1D:
    """
    1D Cubic Spline class

    Parameters
    ----------
    x : list
        x coordinates for data points. This x coordinates must be
        sorted
        in ascending order.
    y : list
        y coordinates for data points

    Examples
    --------
    You can interpolate 1D data points.
    .. image:: cubic_spline_1d.png

    """

    def __init__(self, x, y):

        h = np.diff(x)
        if np.any(h < 0):
            raise ValueError("x coordinates must be sorted in ascending order")

        self.a, self.b, self.c, self.d = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)  # dimension of x

        # calc coefficient a
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h, self.a)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            d = (self.c[i + 1] - self.c[i]) / (3.0 * h[i])
            b = 1.0 / h[i] * (self.a[i + 1] - self.a[i]) \
                - h[i] / 3.0 * (2.0 * self.c[i] + self.c[i + 1])
            self.d.append(d)
            self.b.append(b)

    def calc_position(self, x):
        """
        Calc `y` position for given `x`.

        if `x` is outside the data point's `x` range, return None.

        Parameters
        ----------
        x : float
            x position to calculate y.

        Returns
        -------
        y : float
            y position for given x.
        """
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        position = self.a[i] + self.b[i] * dx + \
                   self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return position

    def calc_first_derivative(self, x):
        """
        Calc first derivative at given x.

        if x is outside the input x, return None

        Parameters
        ----------
        x : float
            x position to calculate first derivative.

        Returns
        -------
        dy : float
            first derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        dy = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return dy

    def calc_second_derivative(self, x):
        """
        Calc second derivative at given x.

        if x is outside the input x, return None

        Parameters
        ----------
        x : float
            x position to calculate second derivative.

        Returns
        -------
        ddy : float
            second derivative for given x.
        """

        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dx = x - self.x[i]
        ddy = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return ddy

    def calc_third_derivative(self, x):
        """
        Calc third derivative at given x.

        if x is outside the input x, return None

        Parameters
        ----------
        x : float
            x position to calculate third derivative.

        Returns
        -------
        dddy : float
            third derivative for given x.
        """
        if x < self.x[0]:
            return None
        elif x > self.x[-1]:
            return None

        i = self.__search_index(x)
        dddy = 6.0 * self.d[i]
        return dddy

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def __calc_B(self, h, a):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] \
                       - 3.0 * (a[i + 1] - a[i]) / h[i]
        return B


class CubicSpline2D:

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = CubicSpline1D(self.s, x)
        self.sy = CubicSpline1D(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        x : float
            x position for given s.
        y : float
            y position for given s.
        """
        x = self.sx.calc_position(s)
        y = self.sy.calc_position(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_curvature_rate(self, s):
        """
        calc curvature rate

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature rate for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        dddx = self.sx.calc_third_derivative(s)
        dddy = self.sy.calc_third_derivative(s)
        a = dx * ddy - dy * ddx
        b = dx * dddy - dy * dddx
        c = dx * ddx + dy * ddy
        d = dx * dx + dy * dy
        return (b * d - 3.0 * a * c) / (d * d * d)

    def calc_yaw(self, s):
        """
        calc yaw

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        yaw : float
            yaw angle (tangent vector) for given s.
        """
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        yaw = math.atan2(dy, dx)
        return yaw


# Default Parameters

LATERAL_MOVEMENT = LateralMovement.HIGH_SPEED
LONGITUDINAL_MOVEMENT = LongitudinalMovement.VELOCITY_KEEPING

MAX_SPEED = 130.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 10.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 2.0  # maximum curvature [1/m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
N_S_SAMPLE = 1  # sampling number of target speed

# cost weights
K_J = 0.1
K_T = 0.1
K_S_DOT = 1.0
K_D = 1.0
K_S = 1.0
K_LAT = 1.0
K_LON = 1.0

SIM_LOOP = 500
show_animation = True


if LATERAL_MOVEMENT == LateralMovement.LOW_SPEED:
    MAX_ROAD_WIDTH = 1.0  # maximum road width [m]
    D_ROAD_W = 0.2  # road width sampling length [m]
    TARGET_SPEED = 3.0 / 3.6  # maximum speed [m/s]
    D_T_S = 0.5 / 3.6  # target speed sampling length [m/s]
    # Waypoints
    WX = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    WY = [0.0, 0.0, 1.0, 0.0, -1.0, -2.0]
    OBSTACLES = np.array([[3.0, 1.0], [5.0, -0.0], [6.0, 0.5], [8.0, -1.5]])
    ROBOT_RADIUS = 0.5  # robot radius [m]

    # Initial state parameters
    INITIAL_SPEED = 1.0 / 3.6  # current speed [m/s]
    INITIAL_ACCEL = 0.0  # current acceleration [m/ss]
    INITIAL_LAT_POSITION = 0.5  # current lateral position [m]
    INITIAL_LAT_SPEED = 0.0  # current lateral speed [m/s]
    INITIAL_LAT_ACCELERATION = 0.0  # current lateral acceleration [m/s]
    INITIAL_COURSE_POSITION = 0.0  # current course position

    ANIMATION_AREA = 5.0  # Animation area length [m]

    STOP_S = 4.0  # Merge and stop distance [m]
    D_S = 0.3  # Stop point sampling length [m]
    N_STOP_S_SAMPLE = 3  # Stop point sampling number
else:
    MAX_ROAD_WIDTH = 7.5  # maximum road width [m]
    D_ROAD_W = 1.0  # road width sampling length [m]
    TARGET_SPEED = 100.0 / 3.6  # target speed [m/s]
    D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
    # Waypoints
    WX = [
    0,  50, 100, 150, 200, 240, 270, 290, 300, 295, 280, 260, 230,
    200, 180, 160, 140, 120, 100,  80,  60,  40,  20,   0, -10, -20,
   -30, -30, -20,  -5,   0
]
    WY = [
    0,   0,   0,   0,   10,  30,  60, 100, 140, 160, 180, 190, 195,
  190, 180, 160, 140, 120, 100,  80,  60,  40,  20,   0, -10, -30,
  -60, -80, -90, -95, -100
]
    # Step 1: Compute arclengths
    s = np.zeros(len(WX))
    s[1:] = np.cumsum(np.sqrt(np.diff(WX) ** 2 + np.diff(WY) ** 2))
    s_max = s[-1]  # the length of the track

    # Step 2: Create cubic spline interpolators
    x_spline = CubicSpline(s, WX)
    y_spline = CubicSpline(s, WY)

    # Step 3: Get the first and second derivative of the parameteriation
    dx_ds = x_spline.derivative(1)
    d2x_ds2 = x_spline.derivative(2)
    dy_ds = y_spline.derivative(1)
    d2y_ds2 = y_spline.derivative(2)

    ## Generate interpolated values for plotting
    s_query = np.linspace(0, s_max, 500)
    x_interp = x_spline(s_query)
    y_interp = y_spline(s_query)

    # output parameterisation and curvature as a casadi instance
    x_ref = ca.interpolant('x_ref', 'linear', [s_query], x_interp)
    y_ref = ca.interpolant('y_ref', 'linear', [s_query], y_interp)
    kappa_ref_ = dx_ds(s_query) * d2y_ds2(s_query) - dy_ds(s_query) * d2x_ds2(s_query)
    kappa_ref = ca.interpolant('kappa', 'linear', [s_query], kappa_ref_)
    # s_query, x_interp, y_interp, dx_ds, dy_ds

    # Track half-width
    r = 7.5  # e.g., 5 meters width on each side

    # Compute derivatives at query points
    dx = dx_ds(s_query)
    dy = dy_ds(s_query)

    # Normalize tangent vectors to unit vectors
    tangent_norm = np.sqrt(dx ** 2 + dy ** 2)
    tangent_unit_x = dx / tangent_norm
    tangent_unit_y = dy / tangent_norm

    # Compute normal vectors (rotated tangent vectors by 90 degrees)
    normal_x = -tangent_unit_y
    normal_y = tangent_unit_x

    # Compute left and right boundary points
    x_left = x_interp + r * normal_x
    y_left = y_interp + r * normal_y

    x_right = x_interp - r * normal_x
    y_right = y_interp - r * normal_y
    # Obstacle list
    OBSTACLES = np.array([
        [30, 0, 2.0],
        [80, 0, 2.0]
    ])

    ROBOT_RADIUS = 2.0  # robot radius [m]

    # Initial state parameters
    INITIAL_SPEED = 10.0 / 3.6  # current speed [m/s]
    INITIAL_ACCEL = 0.0  # current acceleration [m/ss]
    INITIAL_LAT_POSITION = 2.0  # current lateral position [m]
    INITIAL_LAT_SPEED = 0.0  # current lateral speed [m/s]
    INITIAL_LAT_ACCELERATION = 0.0  # current lateral acceleration [m/s]
    INITIAL_COURSE_POSITION = 0.0  # current course position

    ANIMATION_AREA = 20.0  # Animation area length [m]
    STOP_S = 60.0  # Merge and stop distance [m]
    D_S = 2  # Stop point sampling length [m]
    N_STOP_S_SAMPLE = 4  # Stop point sampling number


class LateralMovementStrategy:
    def calc_lateral_trajectory(self, fp, di, c_d, c_d_d, c_d_dd, Ti):
        """
        Calculate the lateral trajectory
        """
        raise NotImplementedError("calc_lateral_trajectory not implemented")

    def calc_cartesian_parameters(self, fp, csp):
        """
        Calculate the cartesian parameters (x, y, yaw, curvature, v, a)
        """
        raise NotImplementedError("calc_cartesian_parameters not implemented")


class HighSpeedLateralMovementStrategy(LateralMovementStrategy):
    def calc_lateral_trajectory(self, fp, di, c_d, c_d_d, c_d_dd, Ti):
        tp = copy.deepcopy(fp)
        s0_d = fp.s_d[0]
        s0_dd = fp.s_dd[0]
        # d'(t) = d'(s) * s'(t)
        # d''(t) = d''(s) * s'(t)^2 + d'(s) * s''(t)
        lat_qp = QuinticPolynomial(
            c_d, c_d_d * s0_d, c_d_dd * s0_d**2 + c_d_d * s0_dd, di, 0.0, 0.0, Ti
        )

        tp.d = []
        tp.d_d = []
        tp.d_dd = []
        tp.d_ddd = []

        # Calculate all derivatives in a single loop to reduce iterations
        for i in range(len(fp.t)):
            t = fp.t[i]
            s_d = fp.s_d[i]
            s_dd = fp.s_dd[i]

            s_d_inv = 1.0 / (s_d + 1e-6) + 1e-6  # Avoid division by zero
            s_d_inv_sq = s_d_inv * s_d_inv  # Square of inverse

            d = lat_qp.calc_point(t)
            d_d = lat_qp.calc_first_derivative(t)
            d_dd = lat_qp.calc_second_derivative(t)
            d_ddd = lat_qp.calc_third_derivative(t)

            tp.d.append(d)
            # d'(s) = d'(t) / s'(t)
            tp.d_d.append(d_d * s_d_inv)
            # d''(s) = (d''(t) - d'(s) * s''(t)) / s'(t)^2
            tp.d_dd.append((d_dd - tp.d_d[i] * s_dd) * s_d_inv_sq)
            tp.d_ddd.append(d_ddd)

        return tp

    def calc_cartesian_parameters(self, fp, csp):
        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            i_kappa = csp.calc_curvature(fp.s[i])
            i_dkappa = csp.calc_curvature_rate(fp.s[i])
            s_condition = [fp.s[i], fp.s_d[i], fp.s_dd[i]]
            d_condition = [
                fp.d[i],
                fp.d_d[i],
                fp.d_dd[i],
            ]
            x, y, theta, kappa, v, a = CartesianFrenetConverter.frenet_to_cartesian(
                fp.s[i], ix, iy, i_yaw, i_kappa, i_dkappa, s_condition, d_condition
            )
            fp.x.append(x)
            fp.y.append(y)
            fp.yaw.append(theta)
            fp.c.append(kappa)
            fp.v.append(v)
            fp.a.append(a)
        return fp


class LowSpeedLateralMovementStrategy(LateralMovementStrategy):
    def calc_lateral_trajectory(self, fp, di, c_d, c_d_d, c_d_dd, Ti):
        s0 = fp.s[0]
        s1 = fp.s[-1]
        tp = copy.deepcopy(fp)
        # d = d(s), d_d = d'(s), d_dd = d''(s)
        # * shift s range from [s0, s1] to [0, s1 - s0]
        lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, s1 - s0)

        tp.d = [lat_qp.calc_point(s - s0) for s in fp.s]
        tp.d_d = [lat_qp.calc_first_derivative(s - s0) for s in fp.s]
        tp.d_dd = [lat_qp.calc_second_derivative(s - s0) for s in fp.s]
        tp.d_ddd = [lat_qp.calc_third_derivative(s - s0) for s in fp.s]
        return tp

    def calc_cartesian_parameters(self, fp, csp):
        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            i_kappa = csp.calc_curvature(fp.s[i])
            i_dkappa = csp.calc_curvature_rate(fp.s[i])
            s_condition = [fp.s[i], fp.s_d[i], fp.s_dd[i]]
            d_condition = [fp.d[i], fp.d_d[i], fp.d_dd[i]]
            x, y, theta, kappa, v, a = CartesianFrenetConverter.frenet_to_cartesian(
                fp.s[i], ix, iy, i_yaw, i_kappa, i_dkappa, s_condition, d_condition
            )
            fp.x.append(x)
            fp.y.append(y)
            fp.yaw.append(theta)
            fp.c.append(kappa)
            fp.v.append(v)
            fp.a.append(a)
        return fp


class LongitudinalMovementStrategy:
    def calc_longitudinal_trajectory(self, c_speed, c_accel, Ti, s0):
        """
        Calculate the longitudinal trajectory
        """
        raise NotImplementedError("calc_longitudinal_trajectory not implemented")

    def get_d_arrange(self, s0):
        """
        Get the d sample range
        """
        raise NotImplementedError("get_d_arrange not implemented")

    def calc_destination_cost(self, fp):
        """
        Calculate the destination cost
        """
        raise NotImplementedError("calc_destination_cost not implemented")


class VelocityKeepingLongitudinalMovementStrategy(LongitudinalMovementStrategy):
    def calc_longitudinal_trajectory(self, c_speed, c_accel, Ti, s0):
        fplist = []
        for tv in np.arange(
            TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S
        ):
            fp = FrenetPath()
            lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)
            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.s = [lon_qp.calc_point(t) for t in fp.t]
            fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
            fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
            fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
            fplist.append(fp)
        return fplist

    def get_d_arrange(self, s0):
        return np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W)

    def calc_destination_cost(self, fp):
        ds = (TARGET_SPEED - fp.s_d[-1]) ** 2
        return K_S_DOT * ds


class MergingAndStoppingLongitudinalMovementStrategy(LongitudinalMovementStrategy):
    def calc_longitudinal_trajectory(self, c_speed, c_accel, Ti, s0):
        if s0 >= STOP_S:
            return []
        fplist = []
        for s in np.arange(
            STOP_S - D_S * N_STOP_S_SAMPLE, STOP_S + D_S * N_STOP_S_SAMPLE, D_S
        ):
            fp = FrenetPath()
            lon_qp = QuinticPolynomial(s0, c_speed, c_accel, s, 0.0, 0.0, Ti)
            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.s = [lon_qp.calc_point(t) for t in fp.t]
            fp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
            fp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
            fp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
            fplist.append(fp)
        return fplist

    def get_d_arrange(self, s0):
        # Only if s0 is less than STOP_S / 3, then we sample the road width
        if s0 < STOP_S / 3:
            return np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W)
        else:
            return [0.0]

    def calc_destination_cost(self, fp):
        ds = (STOP_S - fp.s[-1]) ** 2
        return K_S * ds

LATERAL_MOVEMENT_STRATEGY: LateralMovementStrategy
LONGITUDINAL_MOVEMENT_STRATEGY: LongitudinalMovementStrategy

if LATERAL_MOVEMENT == LateralMovement.HIGH_SPEED:
    LATERAL_MOVEMENT_STRATEGY = HighSpeedLateralMovementStrategy()
else:
    LATERAL_MOVEMENT_STRATEGY = LowSpeedLateralMovementStrategy()

if LONGITUDINAL_MOVEMENT == LongitudinalMovement.VELOCITY_KEEPING:
    LONGITUDINAL_MOVEMENT_STRATEGY = VelocityKeepingLongitudinalMovementStrategy()
else:
    LONGITUDINAL_MOVEMENT_STRATEGY = MergingAndStoppingLongitudinalMovementStrategy()



class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []  # d'(s)
        self.d_dd = []  # d''(s)
        self.d_ddd = []  # d'''(t) in low speed / d'''(s) in high speed
        self.s = []
        self.s_d = []  # s'(t)
        self.s_dd = []  # s''(t)
        self.s_ddd = []  # s'''(t)
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.a = []
        self.ds = []
        self.c = []

    def pop_front(self):
        self.x.pop(0)
        self.y.pop(0)
        self.yaw.pop(0)
        self.v.pop(0)
        self.a.pop(0)
        self.s.pop(0)
        self.s_d.pop(0)
        self.s_dd.pop(0)
        self.s_ddd.pop(0)
        self.d.pop(0)
        self.d_d.pop(0)
        self.d_dd.pop(0)
        self.d_ddd.pop(0)


def calc_frenet_paths(c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    for Ti in np.arange(MIN_T, MAX_T, DT):
        lon_paths = LONGITUDINAL_MOVEMENT_STRATEGY.calc_longitudinal_trajectory(
            c_s_d, c_s_dd, Ti, s0
        )

        for fp in lon_paths:
            for di in LONGITUDINAL_MOVEMENT_STRATEGY.get_d_arrange(s0):
                tp = LATERAL_MOVEMENT_STRATEGY.calc_lateral_trajectory(
                    fp, di, c_d, c_d_d, c_d_dd, Ti
                )

                Jp = sum(np.power(tp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tp.s_ddd, 2))  # square of jerk

                lat_cost = K_J * Jp + K_T * Ti + K_D * tp.d[-1] ** 2
                lon_cost = (
                    K_J * Js
                    + K_T * Ti
                    + LONGITUDINAL_MOVEMENT_STRATEGY.calc_destination_cost(tp)
                )
                tp.cf = K_LAT * lat_cost + K_LON * lon_cost
                frenet_paths.append(tp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    return [
        LATERAL_MOVEMENT_STRATEGY.calc_cartesian_parameters(fp, csp) for fp in fplist
    ]


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        ox, oy, r = ob[i, 0], ob[i, 1], ob[i, 2]
        d = [
            ((ix - ox)**2 + (iy - oy)**2)
            for (ix, iy) in zip(fp.x, fp.y)
        ]
        collision = any([di <= (ROBOT_RADIUS + r)**2 for di in d])
        if collision:
            return False
    return True



def check_paths(fplist, ob):
    path_dict = {
        "max_speed_error": [],
        "max_accel_error": [],
        "max_curvature_error": [],
        "collision_error": [],
        "ok": [],
    }
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].v]):  # Max speed check
            path_dict["max_speed_error"].append(fplist[i])
        elif any([abs(a) > MAX_ACCEL for a in fplist[i].a]):  # Max accel check
            path_dict["max_accel_error"].append(fplist[i])
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
            path_dict["max_curvature_error"].append(fplist[i])
        elif not check_collision(fplist[i], ob):
            path_dict["collision_error"].append(fplist[i])
        else:
            path_dict["ok"].append(fplist[i])
    return path_dict


def frenet_optimal_planning(csp, s0, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fpdict = check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fpdict["ok"]:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return [best_path, fpdict]


def generate_target_course(x, y):
    csp = CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def main():
    print(__file__ + " start!!")

    tx, ty, tyaw, tc, csp = generate_target_course(WX, WY)

    # Initialize state using global parameters
    c_s_d = INITIAL_SPEED
    c_s_dd = INITIAL_ACCEL
    c_d = INITIAL_LAT_POSITION
    c_d_d = INITIAL_LAT_SPEED
    c_d_dd = INITIAL_LAT_ACCELERATION
    s0 = INITIAL_COURSE_POSITION

    area = ANIMATION_AREA

    last_path = None

    for i in range(SIM_LOOP):
        [path, fpdict] = frenet_optimal_planning(
            csp, s0, c_s_d, c_s_dd, c_d, c_d_d, c_d_dd, OBSTACLES
        )

        if path is None:
            path = copy.deepcopy(last_path)
            path.pop_front()
        if len(path.x) <= 1:
            print("Finish")
            break

        last_path = path
        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_s_d = path.s_d[1]
        c_s_dd = path.s_dd[1]
        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )


            plt.plot(tx, ty, "k--", label="Center Line")
            plt.plot(x_left, y_left, "g--", linewidth=1.2, label="Left Boundary")
            plt.plot(x_right, y_right, "r--", linewidth=1.2, label="Right Boundary")
            # plt.plot(OBSTACLES[:, 0], OBSTACLES[:, 1], "xk", label="Obstacles")
            for obs in OBSTACLES:
                obs_x, obs_y, obs_r = obs
                circle = patches.Circle((obs_x, obs_y), radius=obs_r, color='black', alpha=0.4)
                plt.gca().add_patch(circle)
                # 可选：显示圆心
                plt.plot(obs_x, obs_y, "kx")
            plt.plot(path.x[1:], path.y[1:], "-ob", label="Planned Path")


            ego_x = path.x[1]
            ego_y = path.y[1]
            ego_yaw = path.yaw[1]
            L = 4.0
            W = 2.0


            arrow_length = 2.5
            plt.arrow(ego_x, ego_y,
                      arrow_length * np.cos(ego_yaw),
                      arrow_length * np.sin(ego_yaw),
                      head_width=1.0,
                      head_length=1.5,
                      fc='c', ec='c')


            cos_yaw = np.cos(ego_yaw)
            sin_yaw = np.sin(ego_yaw)
            rear_to_center = L * 0.5
            corner_offsets = np.array([
                [rear_to_center, W / 2],
                [rear_to_center, -W / 2],
                [-rear_to_center, -W / 2],
                [-rear_to_center, W / 2]
            ])
            R = np.array([[cos_yaw, -sin_yaw],
                          [sin_yaw, cos_yaw]])
            rect = (R @ corner_offsets.T).T + [ego_x, ego_y]

            plt.fill(rect[:, 0], rect[:, 1], facecolor='cyan', edgecolor='blue', alpha=0.5, label="Ego Vehicle")

            zoom_area = 30.0
            plt.xlim(ego_x - zoom_area, ego_x + zoom_area)
            plt.ylim(ego_y - zoom_area, ego_y + zoom_area)

            plt.title("v[km/h]: " + str(path.v[1] * 3.6)[0:4])
            plt.grid(True)
            plt.legend()
            plt.pause(0.0001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == "__main__":
    main()
