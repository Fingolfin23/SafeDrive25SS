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
# Dynamic Vehicle Model (Double Track)
# ========================================
# Author: Ke Xin
# Date: 2025-05-12

# ========================================


def calc_tire_force(alpha, model="nonlinear", c_alpha=3200.0, F_z=3750.0,
                    C1=1.3, C2=15.0, C3=0.5):
    """
    Calculate lateral tire force based on the selected tire model.

    Parameters:
        alpha (float): Slip angle (radians)
        model (str): 'linear' or 'nonlinear'
        c_alpha (float): Cornering stiffness [N/rad] for linear model
        F_z (float): Vertical load [N] for nonlinear model
        C1, C2, C3 (float): Nonlinear tire model parameters

    Returns:
        float: Lateral force [N]
    """
    if model == "linear":
        return c_alpha * alpha
    elif model == "nonlinear":
        S_res = np.sqrt(2 - 2 * np.cos(alpha))
        if S_res < 1e-6:
            return 0.0
        mu_res = C1 * (1 - np.exp(-C2 * S_res)) - C3 * S_res
        return mu_res * F_z  # S_Y/S_res ≈ 1 assumption
    else:
        raise ValueError("Unknown tire model type: choose 'linear' or 'nonlinear'.")


class DynamicModel_DoubleTrack:
    def __init__(self, X=0.0, Y=0.0, Psi=0.0, v=0.01, beta=0.0, Psi_dot=0.0,
                 l_f=1.2, l_r=1.6, w_f=1.5, w_r=1.5,
                 m=1500.0, Jzz=2250.0,
                 c_alpha_f=3200.0, c_alpha_r=3400.0,
                 dt=0.01, tire_model="linear"):
        """
        Double-Track Vehicle Model (DTM) with flexible tire model (linear or nonlinear).
        State: [X, Y, Ψ, v, β, Ψ̇]
        """
        # State variables
        self.X = X
        self.Y = Y
        self.Psi = Psi
        self.v = v
        self.beta = beta
        self.Psi_dot = Psi_dot

        # Vehicle parameters
        self.l_f = l_f
        self.l_r = l_r
        self.w_f = w_f
        self.w_r = w_r
        self.m = m
        self.Jzz = Jzz
        self.c_alpha_f = c_alpha_f
        self.c_alpha_r = c_alpha_r

        # Simulation parameters
        self.dt = dt
        self.g = 9.81
        self.tire_model = tire_model

    def update(self, delta_f, F_dr):
        """
        Update vehicle state using selected tire model.

        Parameters:
            delta_f (float): Front wheel steering angle (radians)
            F_dr (float): Longitudinal driving force (N)
        """
        delta_r = 0.0  # No rear steering

        X, Y, Psi = self.X, self.Y, self.Psi
        v, beta, Psi_dot = self.v, self.beta, self.Psi_dot
        l_f, l_r = self.l_f, self.l_r
        w_f, w_r = self.w_f, self.w_r
        m, Jzz = self.m, self.Jzz
        c_alpha_f, c_alpha_r = self.c_alpha_f, self.c_alpha_r

        v = max(v, 1e-3)  # Avoid division by zero

        v_x = v * np.cos(beta)
        v_y = v * np.sin(beta)

        # Slip angles
        alpha_fl = delta_f - np.arctan2(v_y + l_f * Psi_dot, v_x - 0.5 * w_f * Psi_dot)
        alpha_fr = delta_f - np.arctan2(v_y + l_f * Psi_dot, v_x + 0.5 * w_f * Psi_dot)
        alpha_rl = delta_r - np.arctan2(v_y - l_r * Psi_dot, v_x - 0.5 * w_r * Psi_dot)
        alpha_rr = delta_r - np.arctan2(v_y - l_r * Psi_dot, v_x + 0.5 * w_r * Psi_dot)

        # Estimate vertical loads (simple static load distribution)
        F_z = m * self.g / 4.0

        # Lateral tire forces
        Fyf_fl = calc_tire_force(alpha_fl, model=self.tire_model, c_alpha=c_alpha_f, F_z=F_z)
        Fyf_fr = calc_tire_force(alpha_fr, model=self.tire_model, c_alpha=c_alpha_f, F_z=F_z)
        Fyr_rl = calc_tire_force(alpha_rl, model=self.tire_model, c_alpha=c_alpha_r, F_z=F_z)
        Fyr_rr = calc_tire_force(alpha_rr, model=self.tire_model, c_alpha=c_alpha_r, F_z=F_z)

        # State derivatives
        dot_X = v * np.cos(Psi + beta)
        dot_Y = v * np.sin(Psi + beta)
        dot_Psi = Psi_dot

        Fy_front = Fyf_fl + Fyf_fr
        Fy_rear = Fyr_rl + Fyr_rr

        dot_beta = (-Psi_dot +
                    (1 / (m * v)) * (Fy_front * np.cos(delta_f - beta) +
                                     Fy_rear * np.cos(delta_r - beta) -
                                     F_dr * np.sin(beta)))

        dot_Psi_dot = (1 / Jzz) * (
                Fyf_fl * (l_f * np.cos(delta_f) - 0.5 * w_f * np.sin(delta_f)) +
                Fyf_fr * (l_f * np.cos(delta_f) + 0.5 * w_f * np.sin(delta_f)) +
                Fyr_rl * (-l_r * np.cos(delta_r) - 0.5 * w_r * np.sin(delta_r)) +
                Fyr_rr * (-l_r * np.cos(delta_r) + 0.5 * w_r * np.sin(delta_r))
        )

        dot_v = (1 / m) * (
                Fy_front * np.sin(beta - delta_f) +
                Fy_rear * np.sin(beta - delta_r) +
                F_dr * np.cos(beta)
        )

        # Euler integration
        self.X += dot_X * self.dt
        self.Y += dot_Y * self.dt
        self.Psi += dot_Psi * self.dt
        self.beta += dot_beta * self.dt
        self.Psi_dot += dot_Psi_dot * self.dt
        self.v += dot_v * self.dt

        # Clip for numerical stability
        self.v = np.clip(self.v, 1e-3, 50.0)
        self.beta = np.clip(self.beta, -np.pi / 2, np.pi / 2)
        self.Psi_dot = np.clip(self.Psi_dot, -5.0, 5.0)

    def get_state(self):
        """
        Return the current vehicle state.
        """
        return [self.X, self.Y, self.Psi, self.v, self.beta, self.Psi_dot]

    def get_corners(self, l=4.0, w=2.0):
        """
        Get the body corners for visualization.
        """
        return get_corners_base(self.X, self.Y, self.Psi, l, w)

    def get_tire_corners(self, delta_f=0.0, delta_r=0.0,
                         l=4.0, w=2.0, tire_length=0.6, tire_width=0.25):
        """
        Get the tire corners for visualization.
        """
        yaw_vec = np.array([np.cos(self.Psi), np.sin(self.Psi)])
        normal_vec = np.array([-np.sin(self.Psi), np.cos(self.Psi)])
        rear_center = np.array([self.X, self.Y]) - self.l_r * yaw_vec
        front_center = np.array([self.X, self.Y]) + self.l_f * yaw_vec

        def get_rectangle(center, angle):
            corners = np.array([
                [tire_length / 2, tire_width / 2],
                [tire_length / 2, -tire_width / 2],
                [-tire_length / 2, -tire_width / 2],
                [-tire_length / 2, tire_width / 2]
            ])
            R = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            return (R @ corners.T).T + center

        half_track_front = self.w_f / 2
        half_track_rear = self.w_r / 2

        rear_left = get_rectangle(rear_center + half_track_rear * normal_vec, self.Psi + delta_r)
        rear_right = get_rectangle(rear_center - half_track_rear * normal_vec, self.Psi + delta_r)
        front_left = get_rectangle(front_center + half_track_front * normal_vec, self.Psi + delta_f)
        front_right = get_rectangle(front_center - half_track_front * normal_vec, self.Psi + delta_f)

        return [front_left, front_right, rear_left, rear_right]
