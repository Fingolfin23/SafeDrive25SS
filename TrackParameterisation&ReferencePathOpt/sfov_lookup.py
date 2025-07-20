# ========== sfov_lookup.py ==========
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

def build_sfov_np(csv_path: str,
                  wrap: bool = True) -> tuple:
    """
    从 sfov_lookup.csv 构建 sfov(s) 的 NumPy 插值函数。

    Parameters
    ----------
    csv_path : str
        CSV 文件路径，必须包含列 's_m' 和 'sfov_m'
    wrap     : bool
        是否将 s 按闭环取模（适用于环形赛道）

    Returns
    -------
    sfov_fun : callable
        连续函数 sfov(s_query)，支持标量或数组
    s_max    : float
        赛道总弧长，可用于 wrap
    """
    tab = pd.read_csv(csv_path)
    s_nodes   = tab["s_m"].values
    fov_nodes = tab["sfov_m"].values
    s_max = s_nodes[-1]

    cs = CubicSpline(s_nodes, fov_nodes,
                     bc_type="natural", extrapolate=True)

    def sfov_np(s_query):
        s_q = np.asarray(s_query, dtype=float)
        if wrap:
            s_q = np.mod(s_q, s_max)
        return cs(s_q)

    return sfov_np, s_max




'''from sfov_lookup import build_sfov_np

csv_path = "../Laptimeoptimization/sfov_lookup.csv"
sfov_fun, s_max = build_sfov_np(csv_path)

# Usage of fov caculator:
print("sfov(100) =", float(sfov_fun(100)))
print("sfov([0, 50, 100]) =", sfov_fun([0, 50, 100]))'''