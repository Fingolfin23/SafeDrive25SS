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
x = np.array([0, 10, 20, 30, 40])
y = np.array([0, 5, -5, 0, 5])
# Call the function
x_spline, y_spline, s_max=interpolate(x,y)
# Generate interpolated values
s_query = np.linspace(0, s_max, 500)
x_interp = x_spline(s_query)
y_interp = y_spline(s_query)
# Plot the result
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

