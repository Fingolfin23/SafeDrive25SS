import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

def compute_curvature(x, y, s):
    dx_ds = np.gradient(x, s)
    dy_ds = np.gradient(y, s)
    theta = np.arctan2(dy_ds, dx_ds)
    dtheta_ds = np.gradient(theta, s)
    kappa = dtheta_ds
    return theta, kappa

def resample_and_save(csv_file, output_file='resampled_monza.csv', ds=0.5, start=0, end=None):
    data = np.loadtxt(csv_file, delimiter=',')
    if end is not None:
        data = data[start:end]
    else:
        data = data[start:]

    x_ref = data[:, 0]
    y_ref = data[:, 1]
    wr = data[:, 2]
    wl = data[:, 3]

    # === 原始路径计算 ===
    dx = np.diff(x_ref)
    dy = np.diff(y_ref)
    ds_raw = np.hypot(dx, dy)
    s_raw = np.insert(np.cumsum(ds_raw), 0, 0)
    s_max = s_raw[-1]
    theta_raw, kappa_raw = compute_curvature(x_ref, y_ref, s_raw)

    # === 重采样路径 ===
    s_uniform = np.arange(0, s_max, ds)
    if s_uniform[-1] < s_max:
        s_uniform = np.append(s_uniform, s_max)

    fx = interp1d(s_raw, x_ref, kind='cubic')
    fy = interp1d(s_raw, y_ref, kind='cubic')
    fwr = interp1d(s_raw, wr, kind='linear')
    fwl = interp1d(s_raw, wl, kind='linear')

    x_new = fx(s_uniform)
    y_new = fy(s_uniform)
    wr_new = fwr(s_uniform)
    wl_new = fwl(s_uniform)
    theta_new, kappa_new = compute_curvature(x_new, y_new, s_uniform)

    # === 保存为 CSV ===
    # df = pd.DataFrame({
    #     's': s_uniform,
    #     'x': x_new,
    #     'y': y_new,
    #     'theta': theta_new,
    #     'kappa': kappa_new,
    #     'wr': wr_new,
    #     'wl': wl_new
    # })
    # df.to_csv(output_file, index=False)
    # print(f"Resampled path saved to {output_file}")

    # === 可视化 ===
    # 图1：原始路径
    plt.figure(figsize=(6, 5))
    x_left = x_ref - wr * np.sin(theta_raw)
    y_left = y_ref + wr * np.cos(theta_raw)
    x_right = x_ref + wl * np.sin(theta_raw)
    y_right = y_ref - wl * np.cos(theta_raw)
    plt.plot(x_ref, y_ref, 'k--', label='Original Reference Path')
    plt.plot(x_left, y_left, 'gray', linewidth=1)
    plt.plot(x_right, y_right, 'gray', linewidth=1)
    plt.scatter(x_ref, y_ref, s=5, color='red', label='Original Points')
    plt.axis('equal')
    plt.title("Original Path")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 图2：重采样路径
    plt.figure(figsize=(6, 5))
    x_left_new = x_new - wr_new * np.sin(theta_new)
    y_left_new = y_new + wr_new * np.cos(theta_new)
    x_right_new = x_new + wl_new * np.sin(theta_new)
    y_right_new = y_new - wl_new * np.cos(theta_new)
    plt.plot(x_new, y_new, 'b-', label='Resampled Path')
    plt.plot(x_left_new, y_left_new, 'gray', linewidth=1)
    plt.plot(x_right_new, y_right_new, 'gray', linewidth=1)
    plt.axis('equal')
    plt.title("Resampled Path")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 图3：曲率对比
    plt.figure(figsize=(7, 4))
    plt.plot(s_raw, kappa_raw, 'r--', label='Original Curvature')
    plt.plot(s_uniform, kappa_new, 'b-', label='Resampled Curvature')
    plt.title("Curvature Comparison")
    plt.xlabel("s (m)")
    plt.ylabel("Curvature (1/m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


    return s_uniform, x_new, y_new, theta_new, kappa_new, wr_new, wl_new

if __name__ == "__main__":
    resample_and_save('Monza.csv', output_file='resampled_monza.csv', ds=1, start=180, end=210)
