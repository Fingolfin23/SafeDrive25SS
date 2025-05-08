import numpy as np


def get_corners_base(x, y, yaw, l=4.0, w=2.0):
    """
    Compute the corner coordinates of the vehicle's rectangular body.

    Parameters:
        x, y (float): Position of the vehicle (center point).
        yaw (float): Heading angle (in radians).
        l (float): Vehicle length.
        w (float): Vehicle width.

    Returns:
        np.ndarray: Array of shape (4, 2) containing the corner positions.
    """
    corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw), np.cos(yaw)]])
    return (R @ corners.T).T + [x, y]


def get_tire_corners_base(x, y, yaw, delta, l_f, l_r,
                          l=4.0, w=2.0, tire_length=0.6, tire_width=0.25):
    """
    Compute the corner coordinates of each tire for visualization.

    Parameters:
        x, y (float): Position of the vehicle center.
        yaw (float): Heading angle (in radians).
        delta (float): Steering angle (in radians).
        l_f, l_r (float): Distance from CG to front/rear axle.
        l, w (float): Vehicle length and width.
        tire_length, tire_width (float): Dimensions of each tire.

    Returns:
        list[np.ndarray]: List of 4 arrays (each 4x2), one per tire.
    """
    yaw_vec = np.array([np.cos(yaw), np.sin(yaw)])
    normal_vec = np.array([-np.sin(yaw), np.cos(yaw)])
    rear_center = np.array([x, y]) - l_r * yaw_vec
    front_center = np.array([x, y]) + l_f * yaw_vec

    def get_rectangle(center, angle):
        corners = np.array([
            [tire_length / 2, tire_width / 2],
            [tire_length / 2, -tire_width / 2],
            [-tire_length / 2, -tire_width / 2],
            [-tire_length / 2, tire_width / 2]
        ])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        return (R @ corners.T).T + center

    half_track = w / 2 * 0.6
    rear_left = get_rectangle(rear_center + half_track * normal_vec, yaw)
    rear_right = get_rectangle(rear_center - half_track * normal_vec, yaw)
    front_left = get_rectangle(front_center + half_track * normal_vec, yaw + delta)
    front_right = get_rectangle(front_center - half_track * normal_vec, yaw + delta)

    return [front_left, front_right, rear_left, rear_right]

# ========================================
# Kinematic Vehicle Model with Euler / RK4
# ========================================
# Author: Ke Xin
# Date: 2025-05-08
# Description:
#   - Supports Euler and RK4 integration.
#   - Simulates a kinematic bicycle model for control and path tracking.
# ========================================


class KinematicModel:
    def __init__(self, x=0.0, y=0.0, psi=0.0, v=0.0,
                 l_f=1.2, l_r=1.6, mass=1500.0, dt=0.1,
                 method='euler'):
        """
        Initialize the kinematic model.

        Parameters:
            x, y (float): Initial position.
            psi (float): Initial yaw angle (radians).
            v (float): Initial velocity.
            l_f, l_r (float): Distances from CG to front/rear axles.
            mass (float): Vehicle mass (optional).
            dt (float): Timestep for integration.
            method (str): Integration method, either 'euler' or 'rk4'.
        """
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v
        self.l_f = l_f
        self.l_r = l_r
        self.m = mass
        self.dt = dt
        self.method = method.lower()
        self.delta = 0.0

    def compute_beta(self, delta):
        """
        Compute the slip angle beta.

        Parameters:
            delta (float): Steering angle (radians).

        Returns:
            float: Slip angle (radians).
        """
        return np.arctan((self.l_r / (self.l_f + self.l_r)) * np.tan(delta))

    def dynamics(self, state, a, delta):
        """
        Compute the time derivative of the state.

        Parameters:
            state (array): [x, y, psi, v]
            a (float): Acceleration input.
            delta (float): Steering angle input.

        Returns:
            np.ndarray: [dx, dy, dpsi, dv]
        """
        x, y, psi, v = state
        beta = self.compute_beta(delta)
        dx = v * np.cos(psi + beta)
        dy = v * np.sin(psi + beta)
        dpsi = v / self.l_r * np.sin(beta)
        dv = a * np.cos(beta)
        return np.array([dx, dy, dpsi, dv])

    def update(self, a, delta):
        """
        Update the vehicle state using either Euler or RK4 integration.

        Parameters:
            a (float): Acceleration input.
            delta (float): Steering angle input (radians).
        """
        self.delta = delta
        state = np.array([self.x, self.y, self.psi, self.v])

        if self.method == 'euler':
            dx, dy, dpsi, dv = self.dynamics(state, a, delta)
            self.x += dx * self.dt
            self.y += dy * self.dt
            self.psi += dpsi * self.dt
            self.v += dv * self.dt
        elif self.method == 'rk4':
            k1 = self.dynamics(state, a, delta)
            k2 = self.dynamics(state + 0.5 * self.dt * k1, a, delta)
            k3 = self.dynamics(state + 0.5 * self.dt * k2, a, delta)
            k4 = self.dynamics(state + self.dt * k3, a, delta)
            state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.x, self.y, self.psi, self.v = state
        else:
            raise ValueError(f"Unsupported integration method: {self.method}")

    def get_state(self):
        """
        Get the current state of the vehicle.

        Returns:
            list: [x, y, psi, v]
        """
        return [self.x, self.y, self.psi, self.v]

    def get_corners(self, l=4.0, w=2.0):
        """
        Get the rectangle corners of the vehicle body.

        Returns:
            np.ndarray: Array of shape (4, 2)
        """
        return get_corners_base(self.x, self.y, self.psi, l, w)

    def get_tire_corners(self, tire_length=0.6, tire_width=0.25, l=4.0, w=2.0):
        """
        Get the tire rectangles for visualization.

        Returns:
            list[np.ndarray]: One 4x2 array per tire
        """
        return get_tire_corners_base(self.x, self.y, self.psi, self.delta,
                                     self.l_f, self.l_r, l, w, tire_length, tire_width)
