import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import casadi as ca
from vehiclemodel.doubletrack import DynamicModel_DoubleTrack


class ReferencePath:
    def __init__(self):
        self.ref_path = self._generate_elliptical_arc_path()

    def _generate_elliptical_arc_path(self):
        a = 100.0
        b = 60.0
        n_points = 800

        theta = np.linspace(-np.pi / 4, 3 * np.pi / 4, n_points)
        x_raw = a * np.cos(theta)
        y_raw = b * np.sin(theta)
        x = x_raw - x_raw[0]
        y = y_raw - y_raw[0]

        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        heading = np.arctan2(dy, dx)
        kappa = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5

        return np.stack([x, y, heading, kappa], axis=1)

    def track_error(self, x, y, psi):
        d = np.linalg.norm(self.ref_path[:, :2] - [x, y], axis=1)
        min_idx = np.argmin(d)
        dx = self.ref_path[min_idx, 0] - x
        dy = self.ref_path[min_idx, 1] - y
        yaw = self.ref_path[min_idx, 2]
        angle = np.arctan2(dy, dx)
        error = d[min_idx] * (-1 if np.sin(yaw - angle) < 0 else 1)
        return error, self.ref_path[min_idx, 3], yaw

    def get_road_edges(self, width=10.0):
        x, y, theta = self.ref_path[:, 0], self.ref_path[:, 1], self.ref_path[:, 2]
        half_w = width / 2
        left_x = x + half_w * np.cos(theta + np.pi / 2)
        left_y = y + half_w * np.sin(theta + np.pi / 2)
        right_x = x + half_w * np.cos(theta - np.pi / 2)
        right_y = y + half_w * np.sin(theta - np.pi / 2)
        return np.column_stack([left_x, left_y]), np.column_stack([right_x, right_y])


def setup_plot(ax, ref_path):
    left_edge, right_edge = ref_path.get_road_edges()
    road_polygon = np.vstack([left_edge, right_edge[::-1]])
    ax.add_patch(Polygon(road_polygon, closed=True, facecolor='lightgray', edgecolor='black', alpha=0.5))
    ax.plot(ref_path.ref_path[:, 0], ref_path.ref_path[:, 1], 'b--', label='Reference Path')
    ax.set_xlim(-20, 220)
    ax.set_ylim(-20, 160)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()


def build_optimizer(ref_path, N=500, dt=0.2):
    opti = ca.Opti()

    l_r = 1.6
    l_f = 1.2

    X = opti.variable(4, N + 1)
    U = opti.variable(2, N)

    x0 = ref_path.ref_path[0, 0]
    y0 = ref_path.ref_path[0, 1]
    psi0 = ref_path.ref_path[0, 2]
    v0 = 10.0

    opti.subject_to(X[:, 0] == ca.vertcat(x0, y0, psi0, v0))
    opti.set_initial(X, 0)
    opti.set_initial(U, 0)

    Q = np.diag([5.0, 5.0, 1.0, 0.5])
    R = np.diag([0.1, 0.2])

    cost = 0
    road_width = 10.0
    half_width = road_width / 2

    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]
        ref = ref_path.ref_path[min(k, len(ref_path.ref_path) - 1), :]
        pos_ref = ca.vertcat(ref[0], ref[1], ref[2], 10.0)

        beta = ca.atan((l_r / (l_f + l_r)) * ca.tan(uk[1]))
        dx = xk[3] * ca.cos(xk[2] + beta)
        dy = xk[3] * ca.sin(xk[2] + beta)
        dpsi = xk[3] / l_r * ca.sin(beta)
        dv = uk[0] * ca.cos(beta)
        x_next = xk + dt * ca.vertcat(dx, dy, dpsi, dv)
        opti.subject_to(X[:, k + 1] == x_next)

        opti.subject_to(opti.bounded(-2.0, uk[0], 2.0))
        opti.subject_to(opti.bounded(-0.5, uk[1], 0.5))

        cost += ca.mtimes([(xk - pos_ref).T, Q, (xk - pos_ref)]) + ca.mtimes([uk.T, R, uk])

        dx_err = xk[0] - ref[0]
        dy_err = xk[1] - ref[1]
        e_lat = ca.sqrt(dx_err**2 + dy_err**2 + 1e-6)
        e_margin = e_lat - half_width
        cost += 100 * ca.if_else(e_margin > 0, e_margin**2, 0)

        cost += 10 * ca.if_else(xk[3] < 2.0, (2.0 - xk[3])**2, 0)
        cost += 5 * uk[1]**2

        cost += 1e5 * ca.if_else(e_lat > half_width + 0.5, 1.0, 0.0)

    opti.minimize(cost)
    opti.solver('ipopt')
    return opti, X, U


def main():
    ref_path = ReferencePath()
    N = 200
    dt = 0.2
    opti, X, U = build_optimizer(ref_path, N=N, dt=dt)
    sol = opti.solve()

    init_state = sol.value(X[:, 0])
    model = DynamicModel_DoubleTrack(X=init_state[0], Y=init_state[1], Psi=init_state[2], v=init_state[3])

    traj_x = []
    traj_y = []

    for k in range(N):
        a = sol.value(U[0, k])
        delta = sol.value(U[1, k])
        model.update(delta_f=delta, F_dr=model.m * a)
        traj_x.append(model.X)
        traj_y.append(model.Y)

    fig, ax = plt.subplots()
    setup_plot(ax, ref_path)
    ax.plot(traj_x, traj_y, 'r-', linewidth=2, label='Optimized Trajectory')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()