#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:19:42 2025

@author: zhuwuzhe, Yanxing Chen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


class Track:
    def __init__(self, file_path="Nuerburgring.csv"):
        """
        initialize the class

        Args:
            file_path (string): the path of the .csv file containing the track data
        """
        ## READ track data from a .csv file ##
        
        # Set the file path
        self.file_path = file_path
        # Read csv file
        df = pd.read_csv(file_path)                
        # Read the columns
        x = df['# x_m'].values
        y = df['y_m'].values
        w_left = df['w_tr_left_m'].values
        w_right = df['w_tr_right_m'].values
        
        ## Interpolate discrete centre points ##
        
        # Compute arclengths
        s = np.zeros(len(x))
        s[1:] = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
        self.s_max=s[-1] #the length of the track
        
        # Create cubic spline interpolators
        self._x_spline = CubicSpline(s, x)
        self._y_spline = CubicSpline(s, y)
        
        # Get the first and second derivative of the parameteriation
        self._dxds = self._x_spline.derivative(1)
        self._d2xds2 = self._x_spline.derivative(2)
        self._dyds = self._y_spline.derivative(1)
        self._d2yds2 = self._y_spline.derivative(2)
        
    
    def plot_track(self, s0=0, s1=None, num=500):
        """
        plot the track in cartisan coordiante system

        Args:
            s0 (number): starting arclength of the plot
            s1 (number): end arclength of the plot
            num (int): number of points to be plotted
            number or 1D array: the corresponding d2x/ds2
        """
        if s1==None:
            s1 = self.s_max
        ## Generate interpolated values for plotting ##
        s_query = np.linspace(s0, s1, num)
        x_points = self._x_spline(s_query) # x values for plot
        y_points = self._y_spline(s_query) # y values for plot
        
        ## Plot the figure ##
        plt.figure(figsize=(8, 6))
        plt.plot(x_points, y_points, label='Interpolated Track', color='blue')
        plt.axis('equal')
        plt.title('Race Track Centerline via Cubic Spline Interpolation')
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        plt.legend()
        plt.grid(True)
        plt.show()
    ### The following defines functions which return the desired track information ###
    def kappa_ref(self, s_query):
        """
        get kappa_ref of the centre line at s_query

        Args:
            s_query (number or 1D array): the values of arclength
    
        Returns:
            number or 1D array: the corresponding kappa_ref
        """
        # Calculate the kappa_ref evaluated at s_query
        kappa_ref = self._dxds(s_query)*self._d2yds2(s_query)-self._dyds(s_query)*self._d2xds2(s_query)
        # Return the results
        return kappa_ref
    
    def x(self, s_query):
        """
        get x-coordiantes of the centre line at s_query

        Args:
            s_query (number or 1D array): the values of arclength
    
        Returns:
            number or 1D array: the corresponding x coordinates
        """
        return self._x_spline(s_query)
    
    def y(self, s_query):
        """
        get the y-coordiantes of the centre line at s_query

        Args:
            s_query (number or 1D array): the values of arclength
    
        Returns:
            number or 1D array: the corresponding y coordinates
        """
        return self._y_spline(s_query)
    
    def dxds(self, s_query):
        """
        get the dx/ds of the centre line at s_query

        Args:
            s_query (number or 1D array): the values of arclength
    
        Returns:
            number or 1D array: the corresponding dx/ds
        """
        return self._dx_ds(s_query)
    
    def dyds(self, s_query):
        """
        get the dy/ds of the centre line at s_query

        Args:
            s_query (number or 1D array): the values of arclength
    
        Returns:
            number or 1D array: the corresponding dy/ds
        """
        return self._dy_ds(s_query)
    
    def d2xds2(self, s_query):
        """
        get the d2x/ds2 of the centre line at s_query

        Args:
            s_query (number or 1D array): the values of arclength
    
        Returns:
            number or 1D array: the corresponding d2x/ds2
        """
        return self._d2x_ds2(s_query)
    
    def d2yds2(self, s_query):
        """
        get the d2y/ds2 of the centre line at s_query

        Args:
            s_query (number or 1D array): the values of arclength
    
        Returns:
            number or 1D array: the corresponding d2y/ds2
        """
        return self._d2y_ds2(s_query)
    
    # ------------------------------------------------------------------
    #  Field‑of‑View boundary: perpendicular line at the nearest point
    #  that lies m metres straight ahead of the vehicle
    # ------------------------------------------------------------------
    def _fov_scalar(self, s_curr: float, m_lookahead: float = 5.0) -> float:
        """
        Internal helper: compute FOV boundary for a single arclength value.

        Parameters
        ----------
        s_curr : float
            Current arclength position of the vehicle.
        m_lookahead : float, optional
            Look‑ahead distance in metres along the *heading* direction to
            construct the target point.  Default is 5 m.

        Returns
        -------
        float
            Arclength position s_fov where the perpendicular line intersects
            the centre line.
        """
        # Current position and unit tangent
        x0, y0 = self._x_spline(s_curr), self._y_spline(s_curr)
        dx, dy = self._dxds(s_curr),   self._dyds(s_curr)
        tn = np.hypot(dx, dy)
        if tn < 1e-8:            # rare numerical corner
            return s_curr
        tx, ty = dx / tn, dy / tn

        # Target point m metres straight ahead of vehicle
        Px, Py = x0 + tx * m_lookahead, y0 + ty * m_lookahead

        # Distance‑squared from centreline point to target
        def dist2(s):
            dxp = self._x_spline(s) - Px
            dyp = self._y_spline(s) - Py
            return dxp * dxp + dyp * dyp

        # Search only forward along track, up to 2·m_lookahead or track end
        s_upper = min(self.s_max, s_curr + 2.0 * m_lookahead)
        res = minimize_scalar(dist2, bounds=(s_curr, s_upper), method="bounded")
        return float(res.x)

    def FOV(self, s_query, m_lookahead: float = 5.0):
        """
        Return the arclength position(s) where the perpendicular FOV boundary
        intersects the centre line.

        The boundary is defined as the line perpendicular to the centre line
        at the *nearest* point to a target that is `m_lookahead` metres straight
        ahead of the vehicle.

        Parameters
        ----------
        s_query : float or np.ndarray
            Current vehicle arclength position(s).
        m_lookahead : float, optional
            Look‑ahead distance in metres (default 5 m).

        Returns
        -------
        float or np.ndarray
            Arclength position(s) s_fov.
        """
        if np.ndim(s_query) == 0:
            return self._fov_scalar(float(s_query), m_lookahead)
        else:
            s_query = np.asarray(s_query).ravel()
            return np.array([self._fov_scalar(s, m_lookahead) for s in s_query])
    
    
        
        
        
        
        