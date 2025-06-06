import numpy as np
import math
import casadi as ca
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import argparse

import sys
sys.path.append('..')

import obstacle_avoidance.geometric_primitives as gp
import obstacle_avoidance.obstacles as obstcl
import obstacle_avoidance.obstacle_aware_track_builder as oatb


MONZA_REF_PATH_FILENAME = 'Monza.csv'
MONZA_OBSTACLES_FILENAME = 'MonzaObstacles.csv'
NUERBURGRING_REF_PATH_FILENAME = 'Nuerburgring.csv'
NUERBURGRING_OBSTACLES_FILENAME = 'NuerburgringObstacles.csv'


class Path:
    def __init__(self, path: np.ndarray):
        """
        Path holder class.
        Params:
          - path - np.ndarray of shape (N, 4),
            path[:, 0] - x coordinates of the central line of the path
            path[:, 1] - y coordinates of the central line of the path
            path[:, 2] - distance to the right bound (m)
            path[:, 3] - distance to the left bound (m)
        """
        self.path = path
        self.x_coordinates = path[:, 0]
        self.y_coordinates = path[:, 1]
        self.dist_to_right_bound = path[:, 2]
        self.dist_to_left_bound = path[:, 3]
        self.min_dist_to_bound = min(
            self.dist_to_right_bound.min(),
            self.dist_to_left_bound.min()
        )
        assert self.min_dist_to_bound > gp.EPS, 'distance to bound has to be positive'

    def get_path_coordnates(self):
        return np.vstack([self.x_coordinates, self.y_coordinates]).T


class Vehicle:
    """
    Required vehicle params holder class
    """
    def __init__(self, length: float, width: float):
        """
        Params:
          - length - vehicle length (m)
          - width - vehicle width (m)
        """
        assert length > gp.EPS, 'vehicle length has to be positive'
        assert width > gp.EPS, 'vehicle width has to be positive'

        self.length = length
        self.width = width


class Config:
    """
    Technical params holder class
    """
    def __init__(self, extra_discretization_points_number):
        self.extra_discretization_points_number = extra_discretization_points_number


# 车辆模型
def vehicle_dynamics(x, u, kappa):
    v    = x[0]
    beta = x[1]
    omega= x[2]
    n    = x[3]
    xi   = x[4]
    delta= u[0]
    F_dr = u[1]
    # 参数
    m = 1200.0
    Jzz = 1260.0
    lf, lr = 1.5, 1.4
    wf, wr_ = 1.6, 1.5
    g = 9.81
    Fz = m * g / 4
    delta_r = 0.0
    epsilon = 1e-4
    mu_roll = 0.015
    rho = 1.2041
    Cd = 0.3
    Af = 1

    def slip_angle(delta, beta, l, w, omega, is_front):
        dy = v * ca.sin(beta) + l * omega if is_front else v * ca.sin(beta) - l * omega
        dx_left = v * ca.cos(beta) - w / 2 * omega
        dx_right = v * ca.cos(beta) + w / 2 * omega
        return delta - ca.atan2(dy, dx_left), delta - ca.atan2(dy, dx_right)
    alpha_fl, alpha_fr = slip_angle(delta, beta, lf, wf, omega, True)
    alpha_rl, alpha_rr = slip_angle(delta_r, beta, lr, wr_, omega, False)

    def Fy(alpha):
        return ca.sin(1.9 * ca.atan(10 * ca.sin(alpha))) + 0.97 * ca.sin(ca.atan(10 * ca.sin(alpha)))
    FY_fl = Fy(alpha_fl)
    FY_fr = Fy(alpha_fr)
    FY_rl = Fy(alpha_rl)
    FY_rr = Fy(alpha_rr)

    SF = (1 - n * kappa) / (v * ca.cos(xi + beta) + epsilon)
    F_roll = mu_roll * m * g
    F_aero = 0.5 * rho * Cd * Af * v**2

    dot_n = SF * v * ca.sin(xi + beta)
    dot_xi = SF * omega - kappa
    dot_beta = SF * (-omega + 1 / (m * v + epsilon) *
                    ((FY_fl + FY_fr) * ca.cos(delta - beta) +
                     (FY_rl + FY_rr) * ca.cos(delta_r - beta) -
                     F_dr * ca.sin(beta)))
    dot_omega = SF / Jzz * (
        FY_fl * (lf * ca.cos(delta) - wf / 2 * ca.sin(delta)) +
        FY_fr * (lf * ca.cos(delta) + wf / 2 * ca.sin(delta)) +
        FY_rl * (-lr * ca.cos(delta_r) - wr_ / 2 * ca.sin(delta_r)) +
        FY_rr * (-lr * ca.cos(delta_r) + wr_ / 2 * ca.sin(delta_r))
    )
    dot_v = SF / m * (
        (FY_fl + FY_fr) * ca.sin(beta - delta) +
        (FY_rl + FY_rr) * ca.sin(beta - delta_r) +
        F_dr * ca.cos(beta) - F_roll - F_aero
    )

    dx = ca.vertcat(dot_v, dot_beta, dot_omega, dot_n, dot_xi)
    return dx, SF, FY_fl, FY_fr, FY_rl, FY_rr


def adapt_central_path_from_dataset(
    ref_path: Path,
    obstacles: list,
    vehicle: Vehicle,
    config: Config,
):
    new_central_path, new_distance_to_path_bound = oatb.ObstacleAwareTrackBuilder(
        obstacles
    ).rebuild_central_path(
        central_path=ref_path.get_path_coordnates(),
        distance_to_path_bound=ref_path.min_dist_to_bound,
        minimal_gap_width=vehicle.width,
        extra_points_number=config.extra_discretization_points_number,
        safe_distance_from_obstacle=0.35,  # 0.25m
        frontal_safe_distance_from_obstacle=vehicle.length+vehicle.width,
        rear_safe_distance_from_obstacle=vehicle.length+vehicle.width,
        is_rectangularized=False,
    )
    tmp = np.asarray([new_distance_to_path_bound] * new_central_path.shape[0])
    return Path(np.vstack([new_central_path.T, tmp, tmp]).T) 


def extract_obstacles(obstacles_filename):
    if obstacles_filename is None:
        return []

    obstacles_raw_list = np.loadtxt(obstacles_filename, delimiter=',').reshape((-1, 3))
    return [
        obstcl.CircleObstacle(gp.Point(item[0], item[1]), item[2]) for item in obstacles_raw_list
    ]


# CSV采样并等间距重采样
def generate_reference_path(
    ref_path_filename,
    obstacles_filename=None,
    ds=0.5,
    max_points=None,
    vehicle: Vehicle = Vehicle(length=5, width=2),
    config: Config = Config(extra_discretization_points_number=50),
):
    data = np.loadtxt(ref_path_filename, delimiter=',')
    if max_points is not None:
        data = data[:max_points]

    init_ref_path = Path(data)
    obstacles = extract_obstacles(obstacles_filename)
    updated_path = adapt_central_path_from_dataset(
        ref_path=init_ref_path,
        obstacles=obstacles,
        vehicle=vehicle,
        config=config,
    )

    dx = np.diff(updated_path.x_coordinates)
    dy = np.diff(updated_path.y_coordinates)
    ds_raw = np.hypot(dx, dy)
    s_raw = np.insert(np.cumsum(ds_raw), 0, 0)
    s_max = s_raw[-1]

    s_uniform = np.arange(0, s_max, ds)
    if s_uniform[-1] < s_max:
        s_uniform = np.append(s_uniform, s_max)

    fx = interp1d(s_raw, updated_path.x_coordinates, kind='cubic')
    fy = interp1d(s_raw, updated_path.y_coordinates, kind='cubic')
    fwr = interp1d(s_raw, updated_path.dist_to_right_bound, kind='linear')
    fwl = interp1d(s_raw, updated_path.dist_to_right_bound, kind='linear')

    x_ref_new = fx(s_uniform)
    y_ref_new = fy(s_uniform)
    wr_new = fwr(s_uniform)
    wl_new = fwl(s_uniform)

    dx_ds = np.gradient(x_ref_new, s_uniform)
    dy_ds = np.gradient(y_ref_new, s_uniform)
    theta = np.arctan2(dy_ds, dx_ds)
    dtheta_ds = np.gradient(theta, s_uniform)
    kappa = dtheta_ds

    new_ref_path = Path(np.vstack([x_ref_new, y_ref_new, wr_new, wl_new]).T)
    return s_uniform, theta, kappa, new_ref_path, init_ref_path, obstacles


def prepare_init_path_boundaries(init_ref_path: Path):
    xs = init_ref_path.x_coordinates
    ys = init_ref_path.y_coordinates
    wls = init_ref_path.dist_to_left_bound
    wrs = init_ref_path.dist_to_right_bound
    n = xs.shape[0]
    x_left, y_left, x_right, y_right = [], [], [], []
    for i in range(1, n + 1):
        cur_direction_v = gp.Vector(
            xs[i % n],
            ys[i % n],
            gp.Point(xs[(i + n - 1) % n], ys[(i + n - 1) % n]),
        )
        left_bound_v = cur_direction_v.rotate_by(math.pi / 2.0).update_length(wls[(i + n - 1) % n])
        right_bound_v = cur_direction_v.rotate_by(-math.pi / 2.0).update_length(wrs[(i + n - 1) % n])
        x_left.append(left_bound_v.x)
        y_left.append(left_bound_v.y)
        x_right.append(right_bound_v.x)
        y_right.append(right_bound_v.y)
    return xs, ys, x_left, y_left, x_right, y_right


def visualize_track(
    X_opt,
    new_ref_path,
    x_opt,
    y_opt,
    theta,
    s,
    obstacles: list,
    init_ref_path: Path,
):
    """
    Params:
      - init_ref_path - reference path from dataset
      - updated_ref_path - reference path with obstacle avoidance
      - optimal_path - optimal speed path
      - obstacles - obstacles on the track
    """
    # === 两个子图 ===
    plt.figure(figsize=(12, 5))

    # 1. velocity profile
    plt.subplot(1, 2, 1)
    plt.plot(s, X_opt[:, 0], label='velocity (m/s)')
    plt.xlabel('s (m)')
    plt.ylabel('velocity (m/s)')
    plt.grid(True)
    plt.title("Velocity Profile")
    plt.legend()

    # 2. trajectory
    fig = plt.subplot(1, 2, 2)
    # obstacles
    for obstacle in obstacles:
        fig.add_patch(
            plt.Circle(
                (obstacle.center.x, obstacle.center.y),
                obstacle.radius,
                alpha=0.5,
            )
        )

    # path
    x_left = new_ref_path.x_coordinates - new_ref_path.dist_to_right_bound * np.sin(theta)
    y_left = new_ref_path.y_coordinates + new_ref_path.dist_to_right_bound * np.cos(theta)
    x_right = new_ref_path.x_coordinates + new_ref_path.dist_to_left_bound * np.sin(theta)
    y_right = new_ref_path.y_coordinates - new_ref_path.dist_to_left_bound * np.cos(theta)
    plt.plot(new_ref_path.x_coordinates, new_ref_path.y_coordinates, 'k--', label='Reference Path')
    plt.plot(x_left, y_left, 'red', linestyle=':', label='Road Edge L')
    plt.plot(x_right, y_right, 'blue', linestyle=':', label='Road Edge R')
    plt.plot(x_opt, y_opt, 'r', label='Optimized Path')
    
    # init path
    x_center, y_center, x_left, y_left, x_right, y_right = prepare_init_path_boundaries(init_ref_path)
    plt.plot(x_left, y_left, 'black', label='Initial Road Edge L')
    plt.plot(x_right, y_right, 'black', label='Initial Road Edge R')
    plt.plot(x_center, y_center, 'green', label='Initial Central Path')

    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title("Trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 优化主程序
def solve_min_time(
    ref_path_filename='Nuerburgring.csv',
    obstacles_filename='NuerburgringObstacles.csv',
    ds=2,
):
    # === 生成等间距参考路径 ===
    s, theta, kappa, new_ref_path, init_ref_path, obstacles = generate_reference_path(
        ref_path_filename=ref_path_filename,
        obstacles_filename=obstacles_filename,
        ds=ds,
        max_points=1160,
    )
    N = len(s) - 1

    # ---- 优化参数 ----
    lambda_ddelta = 10 # 10
    lambda_dF = 0.01
    lambda_dn = 10.0
    lambda_dxi = 5  # 5
    lambda_dv = 1.0
    lambda_jerk = 3.0  # 1

    m = 1500.0
    g = 9.81
    F_z = m * g / 4
    mu = 0.9
    F_x_max = 7100.0 / 2
    F_y_max = mu * F_z
    a_x_max = mu * g
    a_y_max = mu * g

    nx, nu = 5, 2
    x0_val = [10.0, 0.0, 0.0, 0.0, 0.0]

    w, w0, lbw, ubw, g_list, lbg, ubg = [], [], [], [], [], [], []
    J = 0
    Xk = ca.MX.sym("X0", nx)
    w += [Xk]
    w0 += x0_val
    lbw += x0_val
    ubw += x0_val
    delta_prev = ca.MX(0)
    delta_prev_prev = ca.MX(0)
    F_prev = ca.MX(0)
    Xk_prev = Xk
    for k in range(N):
        Uk = ca.MX.sym(f"U_{k}", nu)
        w += [Uk]
        w0 += [0.0, 500.0]
        lbw += [-np.pi/6, -21000.0]
        ubw += [np.pi/6, 7100.0]

        kappa_k = float(kappa[k])
        Xk_next = ca.MX.sym(f"X_{k + 1}", nx)
        w += [Xk_next]
        w0 += [10.0, 0.0, 0.0, 0.0, 0.0]

        # ==== 动态道路宽度约束 ====
        n_min_k = -new_ref_path.dist_to_right_bound[k]
        n_max_k = new_ref_path.dist_to_left_bound[k]

        lbw += [0, -ca.inf, -2, n_min_k, -ca.inf]
        ubw += [42.5, ca.inf, 2, n_max_k, ca.inf]

        f_k, SF_k, FY_fl, FY_fr, FY_rl, FY_rr = vehicle_dynamics(Xk, Uk, kappa_k)
        g_list += [Xk + ds * f_k - Xk_next]
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        J += ds * SF_k
        if k > 0:
            J += lambda_ddelta * (Uk[0] - delta_prev) ** 2
            J += lambda_dF * (Uk[1] - F_prev) ** 2
            J += lambda_dn * (Xk[3] - Xk_prev[3]) ** 2
            J += lambda_dxi * (Xk[4] - Xk_prev[4]) ** 2
            J += lambda_dv * (Xk[0] - Xk_prev[0]) ** 2
        if k > 1:
            J += lambda_jerk * (Uk[0] - 2 * delta_prev + delta_prev_prev) ** 2
        delta_prev_prev = delta_prev
        delta_prev = Uk[0]
        F_prev = Uk[1]
        Xk_prev = Xk

        # ----------- 轮胎椭圆限制 -----------
        F_dr = Uk[1]
        F_x_rl = F_dr / 2
        F_x_rr = F_dr / 2
        ellipse_rl = (F_x_rl / F_x_max) ** 2 + (FY_rl / F_y_max) ** 2
        ellipse_rr = (F_x_rr / F_x_max) ** 2 + (FY_rr / F_y_max) ** 2
        g_list += [ellipse_rl, ellipse_rr]
        lbg += [-ca.inf, -ca.inf]
        ubg += [1.0, 1.0]

        Xk = Xk_next

    nlp = {"f": J, "x": ca.vertcat(*w), "g": ca.vertcat(*g_list)}
    solver = ca.nlpsol("solver", "ipopt", nlp, {"ipopt.print_level": 5, "print_time": False,"ipopt.max_iter": 1000})
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    w_opt = sol['x'].full().flatten()
    X_opt, U_opt = [], []
    for k in range(N):
        base = k * (nx + nu)
        X_opt.append(w_opt[base: base + nx])
        U_opt.append(w_opt[base + nx: base + nx + nu])
    X_opt.append(w_opt[-nx:])
    X_opt = np.array(X_opt)
    U_opt = np.array(U_opt)

    n_opt = X_opt[:, 3]
    x_opt = new_ref_path.x_coordinates - n_opt * np.sin(theta)
    y_opt = new_ref_path.y_coordinates + n_opt * np.cos(theta)

    visualize_track(X_opt, new_ref_path, x_opt, y_opt, theta, s, obstacles, init_ref_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Optimal path',
        description='Calculates optimal path on the track avoiding obstacles',
    )
    parser.add_argument(
        '--track',
        choices=['monza', 'nuerburgring'],
        default='nuerburgring',
        required=False,
    )
    args = parser.parse_args()

    ref_path_filename = None
    obstacles_filename = None
    if args.track == 'monza':
        ref_path_filename = MONZA_REF_PATH_FILENAME
        obstacles_filename = MONZA_OBSTACLES_FILENAME
    else:
        ref_path_filename = NUERBURGRING_REF_PATH_FILENAME
        obstacles_filename = NUERBURGRING_OBSTACLES_FILENAME

    solve_min_time(
        ref_path_filename=ref_path_filename,
        obstacles_filename=obstacles_filename,
        ds=3,
    )
