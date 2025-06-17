#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 14:59:47 2025

@author: zhuwuzhe
"""

import numpy as np
import casadi as ca

def C_opt(car_param, ds, s0, S_fov, kappa_ref, global_sub, x_start):
    """optimizing problem B by casadi

    This function uses casadi to solve problem B

    Args:
        car_param (dic[str, numpy array]): car parameters
        ds (float): step size for arclength s
        s0 (float): initial position projected on center line
        S_fov (float): position of boundary of FOV
        kappa_ref (function): takes a position s on the center line then returns the curvature of the center line at s
        global_sub (dic[str, numpy array]): subarray of the global solution for each states or control
        x_start (dic[str, numpy array]): states at the starting position s0

    Returns:
        opti: an casadi optimizer instance
        n: lateral distance from the center line
        xi: heading error
        beta: sideslip angle
        psi_dot: yaw rate
        v: speed
        delta_f: front steering angle
        F_dr: driving force

    Raises:
        
    """
    ##==unpack car parameters==##
    m = car_param["m"]
    g = car_param["g"]
    mu_roll = car_param["mu_roll"]
    mu = car_param["mu"]
    F_MAX = car_param["F_MAX"]
    F_MIN = car_param["F_MIN"] #breaking force
    delta_MAX = car_param["delta_MAX"]
    J_zz = car_param["J_zz"]
    l_f = car_param["l_f"]
    l_r = car_param["l_r"]
    w_f = car_param["w_f"]
    w_r = car_param["w_r"]
    
    ##==initialize the independent variable (i.e. s)==#
    N = int((S_fov-s0)/ds+1) # number of discretization
    s = np.linspace(s0,S_fov,N)
    # or can take the s from data
    
    ##==create a casadi optimizer instance==##
    opti = ca.Opti()
    
    ##==define optimization variables==##
    n = opti.variable(N)
    xi = opti.variable(N)
    beta = opti.variable(N)
    psi_dot = opti.variable(N)
    v = opti.variable(N)
    delta_f = opti.variable(N-1)
    F_dr = opti.variable(N-1)
    dtds = opti.variable(N)
    
    ##==get the global solution for states and corresponding control from s0 to S_fov==##
    # The following guess is to be followed by the car as a euqlity constraint during opt
    n_guess = global_sub['n']
    # They are used as initial guess for opti
    xi_guess = global_sub['xi']
    beta_guess = global_sub['beta']
    psi_dot_guess = global_sub['psi_dot']
    delta_f_guess = global_sub['delta_f']
    v_guess = global_sub['v']
    F_dr_guess = global_sub['F_dr']
    
    ##==define some constants==##
    F_roll = mu_roll * m * g
    F_Z = m * g / 4
    
    for k in range(N-1):
        kappa = kappa_ref(s[k])
        # slip angle for 4 tyres
        alpha_fl = delta_f[k] - ca.atan2(v[k]*ca.sin(beta[k]) + l_f*psi_dot[k], v[k]*ca.cos(beta[k]) - w_f/2*psi_dot[k])
        alpha_fr = delta_f[k] - ca.atan2(v[k]*ca.sin(beta[k]) + l_f*psi_dot[k], v[k]*ca.cos(beta[k]) + w_f/2*psi_dot[k])
        alpha_rl = 0 - ca.atan2(v[k]*ca.sin(beta[k]) - l_r*psi_dot[k], v[k]*ca.cos(beta[k]) - w_r/2*psi_dot[k])
        alpha_rr = 0 - ca.atan2(v[k]*ca.sin(beta[k]) - l_r*psi_dot[k], v[k]*ca.cos(beta[k]) + w_r/2*psi_dot[k])
        # magical formula for tyre lateral force
        F_y = lambda alpha: ca.sin(1.9 * ca.atan(10 * ca.sin(alpha))) + 0.97 * ca.sin(ca.atan(10 * ca.sin(alpha)))
        FYfl = F_y(alpha_fl)
        FYfr = F_y(alpha_fr)
        FYrl = F_y(alpha_rl)
        FYrr = F_y(alpha_rr)
        # system dynamics
        dn = dtds[k] * (v[k]*ca.sin(xi[k] + beta[k]))
        dxi = dtds[k] * psi_dot[k] - kappa
        dbeta = dtds[k] * (-psi_dot[k] + 1/(m*v[k]) * (
            (FYfl + FYfr)*ca.cos(delta_f[k] - beta[k]) +
            (FYrl + FYrr)*ca.cos(- beta[k]) - F_dr[k]*ca.sin(beta[k])
        ))
        dpsi_dot = dtds[k]/J_zz * (
            FYfl*(l_f*ca.cos(delta_f[k]) - w_f/2*ca.sin(delta_f[k])) +
            FYfr*(l_f*ca.cos(delta_f[k]) + w_f/2*ca.sin(delta_f[k])) +
            FYrl*(-l_r) +
            FYrr*(-l_r)
        )
        dv = dtds[k]/m * (
            (FYfl + FYfr)*ca.sin(beta[k] - delta_f[k]) +
            (FYrl + FYrr)*ca.sin(beta[k]) +
            F_dr[k]*ca.cos(beta[k]) - F_roll
        )
    
        # Integrate
        opti.subject_to(n[k+1] == n[k] + ds*dn)
        opti.subject_to(xi[k+1] == xi[k] + ds*dxi)
        opti.subject_to(beta[k+1] == beta[k] + ds*dbeta)
        opti.subject_to(psi_dot[k+1] == psi_dot[k] + ds*dpsi_dot)
        opti.subject_to(v[k+1] == v[k] + ds*dv)
    
        # Control and friction constraints
        opti.subject_to(F_dr[k] <= F_MAX)
        opti.subject_to(F_dr[k] >= F_MIN)
        opti.subject_to(ca.fabs(delta_f[k]) <= delta_MAX)
    
        opti.subject_to((FYfl/(mu*F_Z))**2 + (0.25*F_roll/(mu*F_Z))**2 <= 1)
        opti.subject_to((FYfr/(mu*F_Z))**2 + (0.25*F_roll/(mu*F_Z))**2 <= 1)
        opti.subject_to((FYrl/(mu*F_Z))**2 + (0.25*(F_dr[k]+F_roll)/(mu*F_Z))**2 <= 1)
        opti.subject_to((FYrr/(mu*F_Z))**2 + (0.25*(F_dr[k]+F_roll)/(mu*F_Z))**2 <= 1)
    
    ##==equality constraint for dtds==##
    for k in range(N):
        opti.subject_to(dtds[k] == (1 - n[k]*kappa_ref(s[k])) / v[k]*ca.cos(xi[k] + beta[k]))
    
    ##==initial state constraint==##
    opti.subject_to(n[0] == x_start["n"])
    opti.subject_to(xi[0] == x_start["xi"])
    opti.subject_to(beta[0] == x_start["beta"])
    opti.subject_to(psi_dot[0] == x_start["psi_dot"])
    opti.subject_to(v[0] == x_start["v"])  ##initial speed when breaking
    
    
    ##==state constraint==##
    
    n_tolerance = 0.1 # meters
    for k in range(N):
        opti.subject_to(v[k] >= 0.1)
        # geometric path following
        # Allow for some small deviation
        opti.subject_to(ca.fabs(n[k] - n_guess[k]) <= n_tolerance)
        #opti.subject_to(xi[k] == xi_guess[k])
        #opti.subject_to(beta[k] == beta_guess[k])
        #opti.subject_to(psi_dot[k] == psi_dot_guess[k])
        
    # F_dr can not be greater than F_dr_guess
    #for k in range(N-1):
    #    opti.subject_to(F_dr[k] <= F_guess[k])
    
    # The first control step is braking
    #opti.subject_to(F_dr[0] <= 0)#F_dr_guess[0])
    
    # The final speed must be smaller than original
    #opti.subject_to(v[N-1] <= v_guess[N-1])
    
    # =========================================================
    # == 新增：添加速度单调递减的约束 ==
    # =========================================================
    #for k in range(N-1):
    #    opti.subject_to(v[k+1] < v[k])
    # =========================================================
    
    
    # Objective
    opti.minimize((100*v[N-1])**2)
    
    opti.set_initial(n, n_guess)
    opti.set_initial(xi, xi_guess)
    opti.set_initial(beta, beta_guess)
    opti.set_initial(psi_dot, psi_dot_guess)
    opti.set_initial(v, v_guess)
    opti.set_initial(delta_f, delta_f_guess)
    opti.set_initial(F_dr, F_dr_guess) # Set the initial guess for the braking force to the negative of the absolute value of the global solution
    
    return opti, n, xi, beta, psi_dot, v, delta_f, F_dr
