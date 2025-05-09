#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track Centerline Interpolation and Visualization using Cubic Splines
--------------------------------------------------------------------

This script demonstrates how to represent and visualize a race track's centerline
as a smooth parametric curve based on discrete sample points. It uses cubic spline
interpolation to map arclength (distance along the track) to 2D Cartesian coordinates.

Steps:
1. Input raw discrete track centerline points (x, y).
2. Compute cumulative arclength values for parameterization.
3. Fit cubic splines to x(s) and y(s) where s is arclength.
4. Evaluate and plot the smooth centerline and original points.

Dependencies:
- numpy
- matplotlib
- scipy

Author: Wuzhe Zhu
Date: 2025.05.08
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import casadi as ca




def interpolate(x,y):
    """
    Given a list of ordered points that discretize a track, the function computes the cublic spline that interpolate them, parameterized by arclength.

    Parameters
    ----------
    x : list
        x coordinates of the discretized track
    y : list
        x coordinates of the discretized track
    

    Returns
    -------
    x_spline : scipy.interpolate.PPoly instance
        a function that maps arclength to the x coordinate of the track in cartisan coordinate system
    y_spline : scipy.interpolate.PPoly instance
        a function that maps arclength to the y coordinate of the track in cartisan coordinate system
    """
    # Step 1: Compute arclengths
    s = np.zeros(len(x))
    s[1:] = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    s_max=s[-1]
    
    # Step 2: Create cubic spline interpolators
    x_spline = CubicSpline(s, x)
    y_spline = CubicSpline(s, y)
    
    return x_spline, y_spline, s_max


# An example

# Input the discretized coordinates
# 中心线坐标（x, y），顺时针排列，单位：米
x = np.array([
    0,  50, 100, 150, 200, 240, 270, 290, 300, 295, 280, 260, 230,
    200, 180, 160, 140, 120, 100,  80,  60,  40,  20,   0, -10, -20,
   -30, -30, -20,  -5,   0
])

y = np.array([
    0,   0,   0,   0,   10,  30,  60, 100, 140, 160, 180, 190, 195,
  190, 180, 160, 140, 120, 100,  80,  60,  40,  20,   0, -10, -30,
  -60, -80, -90, -95, -100
])
# Call the function
x_spline, y_spline, s_max=interpolate(x,y)

# Get the first and second derivative
dx_ds = x_spline.derivative(1)
d2x_ds2 = x_spline.derivative(2)
dy_ds = y_spline.derivative(1)
d2y_ds2 = y_spline.derivative(2)

# Generate interpolated values
s_query = np.linspace(0, s_max, 500)
x_interp = x_spline(s_query)
y_interp = y_spline(s_query)
dx_vals = dx_ds(s_query)
d2x_vals = d2x_ds2(s_query)

# output parameterisation and curvature as a casadi instance
x_ref = ca.interpolant('x_ref', 'linear', [s_query], x_interp)
y_ref = ca.interpolant('y_ref', 'linear', [s_query], y_interp)
kappa_ref_ = dx_ds(s_query)*d2y_ds2(s_query)-dy_ds(s_query)*d2x_ds2(s_query)
kappa_ref = ca.interpolant('kappa', 'linear', [s_query], kappa_ref_)

# 明确导出接口
__all__ = ['kappa_ref', 'x_ref', 'y_ref', 's_max']

# Plot the spline
plt.figure(figsize=(8, 6))
plt.plot(x_interp, y_interp, label='Interpolated Track', color='blue')
plt.plot(x, y, 'ro', label='Original Points')
plt.axis('equal')
plt.title('Race Track Centerline via Cubic Spline Interpolation')
plt.xlabel('x (meters)')
plt.ylabel('y (meters)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the x derivative and curvature
plt.figure(figsize=(10, 6))
plt.plot(s_query, dx_vals, '--', label="x'(s)", linewidth=1.5)
plt.plot(s_query, d2x_vals, ':', label="x''(s)", linewidth=1.5)
plt.plot(s_query, kappa_ref_, ':', label="kappa(s)", linewidth=1.5)

plt.title('Cubic Spline and Its Derivatives')
plt.xlabel('s')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

