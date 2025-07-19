#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 10:33:42 2025

@author: zhuwuzhe
"""
import numpy as np
import casadi as ca

def P_i_opt(car_param, ds, s0, s_end, kappa_ref, global_sub, sol_pre_P, sj, fov, i, x_start):
    """optimizing problem P by casadi

    This function uses casadi to solve problem P

    Args:
        car_param (dic[str, numpy array]): car parameters
        ds (float): step size for arclength s
        s0 (float): initial position projected on center line
        s_end (float): the end point of the segment of track on which optimization works
        kappa_ref (function): takes a position s on the center line then returns the curvature of the center line at s
        sol_pre_P (list of dictionary): sol_pre_P[i]["n"] is the array constaining solution states n of imaginary car i+1.
        global_sub (dic[str, numpy array]): subarray of the global solution for each states or control
        sj (1d array): containing points that safety needs to guarantee
        fov (function R->R): returns the boundary of fov given a point on the track, fov(s0+N*ds)-ds must be a multiple of ds
        i (int): index of the problem(P_i)
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
        n_im(list of casadi.casadi.MX): n_im[i] contains the state n for imaginary car i+1 
        xi_im(list of casadi.casadi.MX): xi_im[i] contains the state xi for imaginary car i+1 
        beta_im(list of casadi.casadi.MX): beta_im[i] contains the state beta for imaginary car i+1 
        psi_dot_im(list of casadi.casadi.MX): psi_dot_im[i] contains the state psi_dot for imaginary car i+1 
        v_im(list of casadi.casadi.MX): v_im[i] contains the state v for imaginary car i+1 
        delta_f_im(list of casadi.casadi.MX): delta_f_im[i] contains the control delta_f for imaginary car i+1 
        F_dr_im(list of casadi.casadi.MX): F_dr_im[i] contains the control F_dr for imaginary car i+1 
        

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
    
    ##==Initialize the independent variable (i.e. s)==#
    N = round((s_end-s0)/ds+1) ## number of discretization
    s = np.linspace(s0,s_end,N)
    # or can take the s from data
    
    ##==creat a casadi optimizer instance==##
    opti = ca.Opti()
    
    ##==define optimization variables of the real car==##
    n = opti.variable(N)
    xi = opti.variable(N)
    beta = opti.variable(N)
    psi_dot = opti.variable(N)
    v = opti.variable(N)
    delta_f = opti.variable(N-1)
    F_dr = opti.variable(N-1)
    dtds = opti.variable(N)
    ##==define optimization variables of the imaginary cars(form car 1 to car i)==##
    n_im=[]
    xi_im=[]
    beta_im=[]
    psi_dot_im=[]
    v_im=[]
    delta_f_im=[]
    F_dr_im=[]
    dtds_im=[]
    N_im=[]
    for l in range(i):
        N_im.append(round((fov(sj[l])-sj[l])/ds+1))
        n_im.append(opti.variable(N_im[l]))
        xi_im.append(opti.variable(N_im[l]))
        beta_im.append(opti.variable(N_im[l]))
        psi_dot_im.append(opti.variable(N_im[l]))
        v_im.append(opti.variable(N_im[l]))
        delta_f_im.append(opti.variable(N_im[l]-1))
        F_dr_im.append(opti.variable(N_im[l]-1))
        dtds_im.append(opti.variable(N_im[l]))
        
        
    
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
    
    ##==adding constraint for the real car==##
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
        
    # equality constraint for dtds
    for k in range(N):
        opti.subject_to(dtds[k] == (1 - n[k]*kappa_ref(s[k])) / v[k]*ca.cos(xi[k] + beta[k]))
    
    ##==initial state constraint==##
    opti.subject_to(n[0] == x_start["n"])
    opti.subject_to(xi[0] == x_start["xi"])
    opti.subject_to(beta[0] == x_start["beta"])
    opti.subject_to(psi_dot[0] == x_start["psi_dot"])
    opti.subject_to(v[0] == x_start["v"])  ##initial speed when breaking
    
    # State constraint, following geometric path from global solution
    n_tolerance=0.1 # meters
    for k in range(N):
        opti.subject_to(v[k] >= 0.1)
        # geometric path following
        opti.subject_to(ca.fabs(n[k] - n_guess[k]) <= n_tolerance)
        #opti.subject_to(xi[k] == xi_guess[k])
        #opti.subject_to(beta[k] == beta_guess[k])
        #opti.subject_to(psi_dot[k] == psi_dot_guess[k])
        
    # F_dr can not be greater than F_dr_guess
    #for k in range(N-1):
    #    opti.subject_to(F_dr[k] <= F_guess[k])
    
    
    ##==adding constraints for the imaginary cars(from car 1 to car i)==##
    for l in range(i):
        # constraint for car l+1
        for k in range(N_im[l]-1):
            kappa = kappa_ref(sj[l]+k*ds)
            # slip angle for 4 tyres
            alpha_fl = delta_f_im[l][k] - ca.atan2(v_im[l][k]*ca.sin(beta_im[l][k]) + l_f*psi_dot_im[l][k], v_im[l][k]*ca.cos(beta_im[l][k]) - w_f/2*psi_dot_im[l][k])
            alpha_fr = delta_f_im[l][k] - ca.atan2(v_im[l][k]*ca.sin(beta_im[l][k]) + l_f*psi_dot_im[l][k], v_im[l][k]*ca.cos(beta_im[l][k]) + w_f/2*psi_dot_im[l][k])
            alpha_rl = 0 - ca.atan2(v_im[l][k]*ca.sin(beta_im[l][k]) - l_r*psi_dot_im[l][k], v_im[l][k]*ca.cos(beta_im[l][k]) - w_r/2*psi_dot_im[l][k])
            alpha_rr = 0 - ca.atan2(v_im[l][k]*ca.sin(beta_im[l][k]) - l_r*psi_dot_im[l][k], v_im[l][k]*ca.cos(beta_im[l][k]) + w_r/2*psi_dot_im[l][k])
            # magical formula for tyre lateral force
            F_y = lambda alpha: ca.sin(1.9 * ca.atan(10 * ca.sin(alpha))) + 0.97 * ca.sin(ca.atan(10 * ca.sin(alpha)))
            FYfl = F_y(alpha_fl)
            FYfr = F_y(alpha_fr)
            FYrl = F_y(alpha_rl)
            FYrr = F_y(alpha_rr)
            
            # system dynamics
            dn = dtds_im[l][k] * (v_im[l][k]*ca.sin(xi_im[l][k] + beta_im[l][k]))
            dxi = dtds_im[l][k] * psi_dot_im[l][k] - kappa
            dbeta = dtds_im[l][k] * (-psi_dot_im[l][k] + 1/(m*v_im[l][k]) * (
                (FYfl + FYfr)*ca.cos(delta_f_im[l][k] - beta_im[l][k]) +
                (FYrl + FYrr)*ca.cos(-beta_im[l][k]) - F_dr_im[l][k]*ca.sin(beta_im[l][k])
            ))
            dpsi_dot = dtds_im[l][k]/J_zz * (
                FYfl*(l_f*ca.cos(delta_f_im[l][k]) - w_f/2*ca.sin(delta_f_im[l][k])) +
                FYfr*(l_f*ca.cos(delta_f_im[l][k]) + w_f/2*ca.sin(delta_f_im[l][k])) +
                FYrl*(-l_r) +
                FYrr*(-l_r)
            )
            dv = dtds_im[l][k]/m * (
                (FYfl + FYfr)*ca.sin(beta_im[l][k] - delta_f_im[l][k]) +
                (FYrl + FYrr)*ca.sin(beta_im[l][k]) +
                F_dr_im[l][k]*ca.cos(beta_im[l][k]) - F_roll
            )
        
            # Integrate
            opti.subject_to(n_im[l][k+1] == n_im[l][k] + ds*dn)
            opti.subject_to(xi_im[l][k+1] == xi_im[l][k] + ds*dxi)
            opti.subject_to(beta_im[l][k+1] == beta_im[l][k] + ds*dbeta)
            opti.subject_to(psi_dot_im[l][k+1] == psi_dot_im[l][k] + ds*dpsi_dot)
            opti.subject_to(v_im[l][k+1] == v_im[l][k] + ds*dv)
        
            # Control and friction constraints
            opti.subject_to(F_dr_im[l][k] <= F_MAX)
            opti.subject_to(F_dr_im[l][k] >= F_MIN)
            opti.subject_to(ca.fabs(delta_f_im[l][k]) <= delta_MAX)
        
            opti.subject_to((FYfl/(mu*F_Z))**2 + (0.25*F_roll/(mu*F_Z))**2 <= 1)
            opti.subject_to((FYfr/(mu*F_Z))**2 + (0.25*F_roll/(mu*F_Z))**2 <= 1)
            opti.subject_to((FYrl/(mu*F_Z))**2 + (0.25*(F_dr_im[l][k]+F_roll)/(mu*F_Z))**2 <= 1)
            opti.subject_to((FYrr/(mu*F_Z))**2 + (0.25*(F_dr_im[l][k]+F_roll)/(mu*F_Z))**2 <= 1)
            
        # equality constraint for dtds
        for k in range(N_im[l]):
            opti.subject_to(dtds_im[l][k] == (1 - n_im[l][k]*kappa_ref(sj[l]+k*ds)) / v_im[l][k]*ca.cos(xi_im[l][k] + beta_im[l][k]))
            
        # get the index of s for the position sj[l], i.e. s[index]=sj[l]
        index=round((sj[l]-s0)/ds)
        
        # initial state constraints for car l+1
        opti.subject_to(n_im[l][0] == n[index])
        opti.subject_to(xi_im[l][0] == xi[index])
        opti.subject_to(beta_im[l][0] == beta[index])
        opti.subject_to(psi_dot_im[l][0] == psi_dot[index])
        opti.subject_to(v_im[l][0] == v[index])  ##initial speed when breaking
        # final state constraints for car l+1
        opti.subject_to(v_im[l][-1] <=0.5 )
        
        # state constraint for imaginary car l+1, width of the track is considered constant
        n_max, n_min=10, -10
        for k in range(N_im[l]):
            # car has nonzero speed
            opti.subject_to(v_im[l][k] >= 0.1)
            # car within the boundary of the track
            opti.subject_to(n_im[l][k]<=n_max)
            opti.subject_to(n_im[l][k]>=n_min)
        
        
    # Objective
    J = ca.sum1(dtds) * ds
    opti.minimize(J)
    
    ##==set initial guess==##
    # real car
    opti.set_initial(n, n_guess)
    opti.set_initial(xi, xi_guess)
    opti.set_initial(beta, beta_guess)
    opti.set_initial(psi_dot, psi_dot_guess)
    opti.set_initial(v, v_guess)
    opti.set_initial(delta_f, delta_f_guess)
    opti.set_initial(F_dr, F_dr_guess) # Set the initial guess for the braking force to the negative of the absolute value of the global solution
    
    # imaginary cars 1,2,...,i-1, starting from previous solutions
    for l in range(i-1):
        opti.set_initial(n_im[l], sol_pre_P[l]["n"])
        opti.set_initial(xi_im[l], sol_pre_P[l]["xi"])
        opti.set_initial(beta_im[l], sol_pre_P[l]["beta"])
        opti.set_initial(psi_dot_im[l], sol_pre_P[l]["psi_dot"])
        opti.set_initial(v_im[l], sol_pre_P[l]["v"])
        opti.set_initial(delta_f_im[l], sol_pre_P[l]["delta_f"])
        opti.set_initial(F_dr_im[l], sol_pre_P[l]["F_dr"])
        
    # imaginary car i, starting  from the global solutions
    # get the index of s for the position sj[i-1], i.e. s[index]=sj[i-1]
    if i>=1:
        index=round((sj[i-1]-s0)/ds)
        opti.set_initial(n_im[i-1], n_guess[index:index+N_im[-1]])
        opti.set_initial(xi_im[i-1], xi_guess[index:index+N_im[-1]])
        opti.set_initial(beta_im[i-1], beta_guess[index:index+N_im[-1]])
        opti.set_initial(psi_dot_im[i-1], psi_dot_guess[index:index+N_im[-1]])
        opti.set_initial(v_im[i-1], v_guess[index:index+N_im[-1]])
        opti.set_initial(delta_f_im[i-1], delta_f_guess[index:index+N_im[-1]-1])
        opti.set_initial(F_dr_im[i-1], F_dr_guess[index:index+N_im[-1]-1])
    
    
    return opti, n, xi, beta, psi_dot, v, delta_f, F_dr, n_im, xi_im, beta_im, psi_dot_im, v_im, delta_f_im, F_dr_im

    
    
    
    