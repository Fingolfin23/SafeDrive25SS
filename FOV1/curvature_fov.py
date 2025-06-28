#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monza - FOV calculation  (V12.1 – final corrected method)
author : Yanxing Chen
date   : 2025-06-28
"""

# ────────────────────────────────
# 0.  Imports
# ────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import casadi as ca     # only needed if you want CasADi interpolants

# ────────────────────────────────
# 1.  Load track CSV
#     (update the path & column names)
# ────────────────────────────────
file_path = "../Laptimeoptimization/Monza.csv"

cols = {
    "x"      : "# x_m",
    "y"      : "y_m",
    "w_left" : "w_tr_left_m",
    "w_right": "w_tr_right_m"
}

df = pd.read_csv(file_path)

x        = df[cols["x"]].to_numpy()
y        = df[cols["y"]].to_numpy()
w_left   = df[cols["w_left"]].to_numpy()
w_right  = df[cols["w_right"]].to_numpy()

# centre-line arc-length
s = np.zeros_like(x)
s[1:] = np.cumsum(np.hypot(np.diff(x), np.diff(y)))
s_max = s[-1]


# ────────────────────────────────
# 2.  Geometry splines  (x(s), y(s))
# ────────────────────────────────
x_spline  = CubicSpline(s, x, bc_type="periodic")
y_spline  = CubicSpline(s, y, bc_type="periodic")

dx_ds   = x_spline.derivative(1)
d2x_ds2 = x_spline.derivative(2)
dy_ds   = y_spline.derivative(1)
d2y_ds2 = y_spline.derivative(2)


# ────────────────────────────────
# 3.  Helper – build_dense_boundaries
# ────────────────────────────────
def build_boundaries(spline_x, spline_y, w_L, w_R, s_dense):
    """Return dense left/right boundary coordinates for a list of s."""
    xc,  yc  = spline_x(s_dense), spline_y(s_dense)
    dx,  dy  = spline_x.derivative(1)(s_dense), spline_y.derivative(1)(s_dense)
    norm_t   = np.hypot(dx, dy)
    tx, ty   = dx/norm_t, dy/norm_t
    nx, ny   = -ty, tx                       # left-hand normal

    wL_d  = np.interp(s_dense, s, w_L)       # width samples
    wR_d  = np.interp(s_dense, s, w_R)

    xl, yl = xc + nx*wL_d, yc + ny*wL_d
    xr, yr = xc - nx*wR_d, yc - ny*wR_d
    return (xl, yl), (xr, yr)


# ────────────────────────────────
# 4.  Final FOV algorithm (V12.1)
#     -- simplified & vectorised core
# ────────────────────────────────
def compute_fov_all_final(sx, sy, wL, wR, s_query, s_full, s_max,
                          delta_s=200.0, dense_step=0.5,
                          ang_min_deg=4, peak_min_deg=1,
                          min_dist=8, min_arc=12, curv_min=1e-5):
    """
    Return raw sfov(s) and cumulative-min filtered version.
    The function matches the logic we discussed (left/right scan,
    flip-detect fallback, gap filtering, etc.).
    """

    ang_min   = np.deg2rad(ang_min_deg)
    peak_min  = np.deg2rad(peak_min_deg)

    # dense boundary samples (for all s)
    s_dense = np.arange(0, s_max + dense_step, dense_step)
    (xl, yl), (xr, yr) = build_boundaries(sx, sy, wL, wR, s_dense)

    sfov_raw = np.empty_like(s_query)

    for k, s0 in enumerate(s_query):
        P0 = np.array([sx(s0), sy(s0)])
        t0 = np.array([sx.derivative(1)(s0), sy.derivative(1)(s0)])
        dir_v = t0 / np.linalg.norm(t0)

        # if curvature very small → straight segment shortcut
        kappa_now = dx_ds(s0)*d2y_ds2(s0) - dy_ds(s0)*d2x_ds2(s0)
        if abs(kappa_now) < curv_min:
            sfov_raw[k] = s0 + delta_s
            continue

        # window indices
        idx0 = np.searchsorted(s_dense, s0)
        idx1 = np.searchsorted(s_dense, s0 + delta_s, side="right")
        s_win = s_dense[idx0:idx1]

        def pick_side(xb, yb, want_max):
            # vector from car to boundary samples
            Vx = xb[idx0:idx1] - P0[0]
            Vy = yb[idx0:idx1] - P0[1]
            dist = np.hypot(Vx, Vy)
            dot  = dir_v[0]*Vx + dir_v[1]*Vy
            mask = (dist > min_dist) & (dot > 0)
            if not np.any(mask):
                return np.inf

            Vx, Vy, dot, ss = Vx[mask], Vy[mask], dot[mask], s_win[mask]
            theta = np.arctan2(dir_v[0]*Vy - dir_v[1]*Vx, dot)

            # ---- flip search ----
            dtheta = np.diff(theta)
            sign   = np.sign(dtheta)
            flips  = np.where(sign[:-1]*sign[1:] < 0)[0] + 1
            cand   = flips if flips.size else \
                     [np.argmax(theta) if want_max else np.argmin(theta)]

            # gap filter
            good = []
            for idx in cand:
                left_gap  = abs(theta[idx] - theta[max(idx-1, 0)])
                right_gap = abs(theta[idx] - theta[min(idx+1, len(theta)-1)])
                if max(left_gap, right_gap) >= peak_min:
                    good.append(idx)
            if not good:
                return np.inf

            idx_sel = max(good, key=lambda i: theta[i]) \
                      if want_max else \
                      min(good, key=lambda i: theta[i])
            s_tan = ss[idx_sel]
            return np.inf if s_tan - s0 < min_arc else s_tan

        s_left  = pick_side(xl, yl, True)
        s_right = pick_side(xr, yr, False)
        sfov_raw[k] = min(s_left, s_right, s0 + delta_s)

    # cumulative min filter (monotone non-increasing look-ahead)
    sfov_filt = np.minimum.accumulate(sfov_raw[::-1])[::-1]
    return sfov_raw, sfov_filt


# ────────────────────────────────
# 5.  Utility – save CSV
# ────────────────────────────────
def save_sfov_csv(s_arr, sfov_arr, path_prefix, out_name):
    csv = pd.DataFrame({"s_m": s_arr, "sfov_m": sfov_arr})
    csv.to_csv(out_name, index=False)
    print(f"sfov written → {out_name}")
    return out_name


# ────────────────────────────────
# 6.  Execution & visualisation (your Part-3)
# ────────────────────────────────
DELTA_S = 200.0            # keep in sync with algorithm above
s_query = np.linspace(0, s_max, 500)

print("Starting FOV calculation with the Final Robust Scanning algorithm (V12.1)...")

sfov_raw_vals, sfov_filtered_vals = compute_fov_all_final(
    x_spline, y_spline, w_left, w_right,
    s_query, s, s_max,
    delta_s       = DELTA_S,
    dense_step    = 0.5,
    ang_min_deg   = 4,
    peak_min_deg  = 1.5,
    min_dist      = 8,
    min_arc       = 12,
    curv_min      = 1e-5
)

save_sfov_csv(s_query, sfov_filtered_vals, file_path,
              "Monza_sfov_V12_final.csv")
print("\nCalculation complete. Visualising results...")

# ---- data for plots ----
look_ahead_dist = sfov_filtered_vals - s_query
kappa_query = dx_ds(s_query)*d2y_ds2(s_query) - dy_ds(s_query)*d2x_ds2(s_query)

# ---- plots ----
fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle('Analysis of Calculated FOV (V12.1 - Final Corrected Method)',
             fontsize=16, weight="bold")

# Track curvature
axs[0].plot(s_query, kappa_query, color='seagreen', lw=1.4,
            label='Curvature (κ)')
axs[0].axhline(0, color='gray', ls='--', lw=0.8)
axs[0].set_ylabel("Curvature κ [1/m]")
axs[0].set_title("Track Curvature", loc="left")
axs[0].grid(True, ls=':', lw=0.6)
axs[0].legend(loc="upper right")

# Look-ahead distance
axs[1].plot(s_query, look_ahead_dist, color='royalblue', lw=1.3,
            label='Look-ahead Distance (sfov − s)')
axs[1].axhline(DELTA_S, color='crimson', ls='--', lw=1.1,
               label=f'Max Look-ahead ({DELTA_S:.0f} m)')
axs[1].set_ylabel("Distance [m]")
axs[1].set_ylim(0, DELTA_S*1.1)
axs[1].set_title("Calculated Look-ahead Distance", loc="left")
axs[1].grid(True, ls=':', lw=0.6)
axs[1].legend(loc="best")

# Raw vs filtered sfov
axs[2].scatter(s_query, sfov_raw_vals, s=6, color='salmon',
               label='Raw sfov (calculated)')
axs[2].plot(s_query, sfov_filtered_vals, color='black', lw=1.6,
            label='Filtered sfov (cumulative min)')
axs[2].plot(s_query, s_query, ls=':', color='gray', lw=1.2,
            label='Current Position (s)')
axs[2].set_xlabel("Arc Length s [m]")
axs[2].set_ylabel("Visible Arc Length sfov [m]")
axs[2].set_title("Raw vs. Filtered sfov Values", loc="left")
axs[2].grid(True, ls=':', lw=0.6)
axs[2].legend(loc="upper left")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()