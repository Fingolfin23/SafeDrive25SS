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
# Dynamic Vehicle Model (Single Track)
# ========================================
# Author: Ke Xin
# Date: 2025-05-08
# Description:
#   - Includes lateral tire dynamics with nonlinear saturation (tanh)
#   - Includes aerodynamic and rolling resistance
#   - Integrates full nonlinear bicycle model with yaw rate and slip angles
# ========================================


class DynamicModel_LinearTire:
    def __init__(self, X=0.0, Y=0.0, phi=0.0, vx=0.01, vy=0.0, omega=0.0,
                 l_f=1.2, l_r=1.6, m=1500.0, Iz=2250.0,
                 Cf=3200.0, Cr=3400.0, dt=0.1,
                 c_r1=0.01, c_a=1.36, consider_drag=True):
        """
        Initialize the dynamic bicycle model.

        Parameters:
            X, Y, phi: initial position and yaw angle (global frame)
            vx, vy: longitudinal and lateral velocity (body frame)
            omega: yaw rate (rad/s)
            l_f, l_r: distances from center of mass to front/rear axles
            m: vehicle mass (kg)
            Iz: yaw moment of inertia (kg*m^2)
            Cf, Cr: cornering stiffness (front and rear)
            dt: simulation time step
            c_r1: rolling resistance coefficient
            c_a: aerodynamic drag coefficient
        """
        self.X = X
        self.Y = Y
        self.phi = phi
        self.vx = vx
        self.vy = vy
        self.omega = omega
        self.l_f = l_f
        self.l_r = l_r
        self.m = m
        self.Iz = Iz
        self.Cf = Cf
        self.Cr = Cr
        self.dt = dt
        self.c_r1 = c_r1
        self.c_a = c_a
        self.delta = 0.0
        self.consider_drag = consider_drag

    def update(self, delta, Fx):
        """
        Update vehicle state using dynamic bicycle model equations.

        Parameters:
            delta (float): steering angle input (rad)
            Fx (float): longitudinal force (N)
        """
        self.delta = np.clip(delta, -np.radians(30.0), np.radians(30.0))  # limit steering
        Fx = np.clip(Fx, -3000.0, 3000.0)  # limit throttle/brake force

        # Global frame position update
        self.X += self.vx * np.cos(self.phi) * self.dt - self.vy * np.sin(self.phi) * self.dt
        self.Y += self.vx * np.sin(self.phi) * self.dt + self.vy * np.cos(self.phi) * self.dt
        self.phi += self.omega * self.dt

        # Compute slip angles (avoid division by 0)
        vx_safe = max(abs(self.vx), 1e-2)
        alpha_f = (self.vy + self.l_f * self.omega) / vx_safe - delta
        alpha_r = (self.vy - self.l_r * self.omega) / vx_safe

        # Nonlinear tire lateral forces using tanh for saturation
        Ffy = -self.Cf * np.tanh(alpha_f)
        Fry = -self.Cr * np.tanh(alpha_r)

        # Longitudinal resistance forces
        if self.consider_drag:
            R_x = self.c_r1 * self.vx  # rolling resistance
            F_aero = self.c_a * self.vx ** 2  # aerodynamic drag
            F_load = F_aero + R_x  # total longitudinal resistance
        else:
            F_load = 0.0

        # Newton's 2nd law in body frame:
        self.vx += ((Fx - Ffy * np.sin(delta) - F_load + self.m * self.vy * self.omega) / self.m) * self.dt
        self.vy += ((Fry + Ffy * np.cos(delta) - self.m * self.vx * self.omega) / self.m) * self.dt
        self.omega += ((Ffy * self.l_f * np.cos(delta) - Fry * self.l_r) / self.Iz) * self.dt

        # Prevent numerical instability
        self.vx = np.clip(self.vx, 1e-3, 50.0)
        self.vy = np.clip(self.vy, -10.0, 10.0)
        self.omega = np.clip(self.omega, -5.0, 5.0)

    def get_state(self):
        """
        Returns:
            list: [X, Y, phi, vx, vy, omega]
        """
        return [self.X, self.Y, self.phi, self.vx, self.vy, self.omega]

    def get_corners(self, l=4.0, w=2.0):
        """
        Returns the body corners for visualization.
        """
        return get_corners_base(self.X, self.Y, self.phi, l, w)

    def get_tire_corners(self, **kwargs):
        """
        Returns corners of each tire for visualization.
        Keyword arguments passed to get_tire_corners_base (e.g. tire_length).
        """
        return get_tire_corners_base(self.X, self.Y, self.phi, self.delta,
                                     self.l_f, self.l_r, **kwargs)
