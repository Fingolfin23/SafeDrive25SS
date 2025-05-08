import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import animation
from scipy.interpolate import CubicSpline
from vehiclemodel.kinematic_model import KinematicModel
from vehiclemodel.dynamic_LinearTire_model import DynamicModel_LinearTire


class PID:
    def __init__(self, Kp=3.0, Ki=0.001, Kd=30.0, target=2.0,
                 upper_force=4000.0, lower_force=-2000.0,
                 max_sum_error=1000.0, alpha=0.2):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = target
        self.upper = upper_force
        self.lower = lower_force
        self.sum_error = 0.0
        self.prev_error = 0.0
        self.max_sum_error = max_sum_error
        self.alpha = alpha  # low-pass filter coefficient
        self.filtered_u = 0.0  # initial filtered output

    def compute(self, current_v):
        error = self.target - current_v
        self.sum_error += error
        self.sum_error = np.clip(self.sum_error, -self.max_sum_error, self.max_sum_error)

        d_error = error - self.prev_error
        self.prev_error = error

        raw_u = self.Kp * error + self.Ki * self.sum_error + self.Kd * d_error
        raw_u = np.clip(raw_u, self.lower, self.upper)

        # First-order low-pass filter
        self.filtered_u = self.alpha * raw_u + (1 - self.alpha) * self.filtered_u

        return self.filtered_u


def geometric_steering_control(e, kappa, psi, ref_psi, l_f, l_r, k_e=0.3, k_psi=1.5):
    """
    Geometric lateral controller: compute steering angle delta.
    """
    heading_error = psi - ref_psi
    heading_error = ((heading_error + np.pi) % (2 * np.pi)) - np.pi
    L = l_f + l_r
    delta = np.arctan(kappa * L) - k_e * e - k_psi * heading_error
    return np.clip(delta, np.radians(-15), np.radians(15))


class ReferencePath:
    def __init__(self):
        self.ref_path = self.generate_custom_path()

    def generate_custom_path(self):
        points = np.array([
            [0, 0], [15, 5], [30, 0], [45, 10], [55, 5],
            [65, 15], [65, 25], [55, 35], [40, 35],
            [25, 35], [10, 35], [-5, 30], [-10, 20],
            [-10, 10], [-5, 0], [0, 0]  # loop back
        ])
        t = np.linspace(0, 1, len(points))
        cs_x = CubicSpline(t, points[:, 0], bc_type='clamped')
        cs_y = CubicSpline(t, points[:, 1], bc_type='clamped')
        t_fine = np.linspace(0, 1, 1000)
        x = cs_x(t_fine)
        y = cs_y(t_fine)
        dx = np.gradient(x)
        dy = np.gradient(y)
        theta = np.arctan2(dy, dx)
        kappa = np.zeros_like(x)  # curvature (can be computed later)
        return np.stack([x, y, theta, kappa], axis=1)

    def track_error(self, robot_state):
        x, y = robot_state[0], robot_state[1]
        d = np.linalg.norm(self.ref_path[:, :2] - [x, y], axis=1)
        min_index = np.argmin(d)
        dx = self.ref_path[min_index, 0] - x
        dy = self.ref_path[min_index, 1] - y
        yaw = self.ref_path[min_index, 2]
        angle = self.normalize_angle(yaw - np.arctan2(dy, dx))
        error = d[min_index] * (-1 if angle < 0 else 1)
        return error, self.ref_path[min_index, 3], yaw, int(min_index)

    def get_road_edges(self, width=6.0):
        x = self.ref_path[:, 0]
        y = self.ref_path[:, 1]
        theta = self.ref_path[:, 2]
        half_w = width / 2
        left_x = x + half_w * np.cos(theta + np.pi / 2)
        left_y = y + half_w * np.sin(theta + np.pi / 2)
        right_x = x + half_w * np.cos(theta - np.pi / 2)
        right_y = y + half_w * np.sin(theta - np.pi / 2)
        return np.column_stack([left_x, left_y]), np.column_stack([right_x, right_y])

    @staticmethod
    def normalize_angle(angle):
        while angle > np.pi: angle -= 2 * np.pi
        while angle < -np.pi: angle += 2 * np.pi
        return angle


def setup_plot(ax, ref_path):
    left_edge, right_edge = ref_path.get_road_edges(width=6.0)
    road_polygon = np.vstack([left_edge, right_edge[::-1]])
    ax.add_patch(Polygon(road_polygon, closed=True, facecolor='lightgray', edgecolor='black', alpha=0.5))
    ax.plot(ref_path.ref_path[:, 0], ref_path.ref_path[:, 1], 'b--', label="Reference Path")
    ax.set_xlim(-20, 70)
    ax.set_ylim(-20, 50)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Vehicle Path Tracking")
    ax.legend()


def create_vehicle_patches(ax):
    car_patch = Polygon([[0, 0]], closed=True, color='red')
    ax.add_patch(car_patch)
    tires = [Polygon([[0, 0]], closed=True, color='black') for _ in range(4)]
    for tire in tires:
        ax.add_patch(tire)
    return car_patch, tires


# ========================================
# demo test
# ========================================
# Author: Ke Xin
# Date: 2025-05-08
# ========================================
# Structure
#     root/
#     │
#     ├── vehiclemodel/                         # Vehicle dynamics and geometry modules
#     │   ├── dynamic_LinearTire_model.py       # Dynamic bicycle model with nonlinear tires and drag
#     │   ├── kinematic_model.py                # Kinematic bicycle model (Euler/RK4)
#     │
#     ├── demo_vehicle_model_test.py            # Main simulation demo with animation and controller
# ========================================
# Environment:
#     Python 3.x
#     numpy==1.21.6
#     matplotlib==3.5.1
#     scipy==1.7.3
#     commonroad-io==2023.1
#     casadi==3.7.0 (optional for MPC extensions)
# ========================================
# Description:
# This script demonstrates path tracking using either a kinematic
# or dynamic vehicle model on a custom spline-based reference path.
# Includes:
# - Geometric lateral controller
# - PID speed controller
# - Matplotlib animation for vehicle motion
# ========================================

def main():
    ref_path = ReferencePath()
    path_xy = ref_path.ref_path[:, :2]
    x0, y0, yaw0 = path_xy[0, 0], path_xy[0, 1], ref_path.ref_path[0, 2]

    # ============ Select Model ============ #
    model_type = "dynamic"  # Options: "kinematic", "dynamic"

    if model_type == "kinematic":
        model = KinematicModel(x=x0, y=y0, psi=yaw0)
        def update_model(m, a, delta): m.update(a, delta)
        get_v = lambda state: state[3]
    elif model_type == "dynamic":
        model = DynamicModel_LinearTire(X=x0, Y=y0, phi=yaw0)
        def update_model(m, a, delta): m.update(delta, Fx=m.m * a)
        def get_v(state): return state[3]
    else:
        raise ValueError("Invalid model_type.")

    # PID controller for longitudinal control
    speed_pid = PID(Kp=200.0, Ki=1.0, Kd=500.0, target=1.0,
                    upper_force=1000.0, lower_force=-500.0)

    fig, ax = plt.subplots()
    setup_plot(ax, ref_path)
    car_patch, tires = create_vehicle_patches(ax)

    states = []
    for _ in range(200):
        state = model.get_state()
        x, y, psi = state[0], state[1], state[2]
        v = get_v(state)

        e, kappa, ref_yaw, _ = ref_path.track_error([x, y, psi])
        delta = geometric_steering_control(e, kappa, psi, ref_yaw, model.l_f, model.l_r)
        a = speed_pid.compute(v)

        update_model(model, a, delta)
        body = model.get_corners()
        tires_xy = model.get_tire_corners()
        states.append((body, tires_xy))

    def update(frame):
        body_xy, tires_xy = states[frame]
        car_patch.set_xy(body_xy)
        for patch, corners in zip(tires, tires_xy):
            patch.set_xy(corners)
        return [car_patch] + tires

    ani = animation.FuncAnimation(fig, update, frames=len(states), interval=30, blit=True)
    plt.show()


if __name__ == "__main__":
    main()
