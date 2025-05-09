#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 13:11:37 2025

@author: zhuwuzhe
"""
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


# 参数
T = 400     # 时间
N = 800     # 时间步数
nx = 4     # 状态维度
nu = 1     # 控制维度
h = T/N    # 时间步长

# define the center line of track
from track_param import kappa_ref, x_ref, y_ref


# define the state and control constraint
kappa_min=-1
kappa_max=1
r_min=-1
r_max=1

# 所有变量
X = ca.MX.sym("X", N * nx)
U = ca.MX.sym("U", (N - 1) * nu)

# 分段变量
# x_list[i] = (s_i,r_i,phi_i,phi_{ref,i})
# u_list[i] = (Kappa_i)
x_list = [X[i*nx:(i+1)*nx] for i in range(N)]
u_list = [U[i*nu:(i+1)*nu] for i in range(N - 1)]

# 目标函数
a1 = 6.5e-4
a2 = 1

obj = -a1*x_list[N-1][0]+a2*(h*sum(u[0] for u in u_list)-h*(u_list[0][0]+u_list[N-2][0])/2)

# 约束列表
g_list = []

# 等式约束：x_{i+1} - x_i - h*u_i = 0
for i in range(N - 1):
    xi = x_list[i]
    ui = u_list[i]
    x_dot = ca.vertcat(
        ca.cos(xi[2]-xi[3])/(1-xi[1]*kappa_ref(xi[0])),
        ca.sin(xi[2]-xi[3]),
        ui[0],
        kappa_ref(xi[0])*ca.cos(xi[2]-xi[3])/(1-xi[1]*kappa_ref(xi[0]))
    )
    xi_next_pred = xi + h * x_dot
    g_list.append(x_list[i + 1] - xi_next_pred)

# 起点条件：x_0 = 0
g_list.append(x_list[0] - [0,0,0,0])

# 不等式约束：
for u in u_list:
    g_list.append(u[0] - kappa_max)     # u <= u_max => u - u_max <= 0
    g_list.append(-u[0] + kappa_min)    # -u <= -u_min => -u + u_min <= 0
for x in x_list:
    g_list.append(x[1] - r_max)
    g_list.append(-x[1] + r_min)

# 连接所有约束
g = ca.vertcat(*g_list)

# 拼接变量
Z = ca.vertcat(X, U)

# NLP问题
nlp = {'x': Z, 'f': obj, 'g': g}
solver = ca.nlpsol('solver', 'ipopt', nlp)

# 初始猜测
z0 = np.zeros(Z.shape[0])

# 约束上下界
n_eq = 4*N -4 + 4     # 动态方程 + 初始条件
n_ineq = 2 * (N - 1) +2*N # 上下限

lbg = [0] * n_eq + [-ca.inf] * n_ineq
ubg = [0] * n_eq + [0] * n_ineq

# 求解
sol = solver(x0=z0, lbg=lbg, ubg=ubg)
z_opt = sol['x'].full().flatten()

# 拆解变量
x_opt = z_opt[:nx*N].reshape((N, nx))
u_opt = z_opt[nx*N:].reshape((N - 1, nu))

print("是否成功:", solver.stats()['success'])
print("最前5个状态:", x_opt[:5])
print("最前5个控制:", u_opt[:5])

# Extract variables
s_vals = x_opt[:, 0]  # x[0]
r_vals = x_opt[:, 1]  # x[1]
phi_vals = x_opt[:,2] # x[2]
phi_ref_vals = x_opt[:,3] # x[3]
x_vals=x_ref(s_vals).full().flatten()-r_vals*np.sin(phi_ref_vals)
y_vals=y_ref(s_vals).full().flatten()+r_vals*np.cos(phi_ref_vals)

# Plot r, phi, phi_ref against s
"""
plt.figure(figsize=(8, 4))
plt.plot(s_vals, r_vals, label='r vs s')
plt.plot(s_vals, phi_vals, label='phi vs s')
plt.plot(s_vals, phi_ref_vals, label='phi_ref_vals vs s')


plt.xlabel('s (x[0])')
plt.ylabel('r (x[1])')
plt.title('Lateral Deviation vs. Arclength')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""
# Plot the curve (x(s),y(s))

plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_vals, label='curve (x,y)')
plt.xlabel('x(s)')
plt.ylabel('y(s)')
plt.title('curve (x,y)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




