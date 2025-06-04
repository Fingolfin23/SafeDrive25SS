#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:19:42 2025

@author: zhuwuzhe
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


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
    
    def FOV(self, s_query):
        """
        get the FOV at s_query
        
        For now the boundary of FOV is considered to be a straight line perpendicular to the center line

        Args:
            s_query (number or 1D array): the values of arclength
    
        Returns:
            number: the arclength position at which the booundary of FOV intersects with the center line
            
        """
        ## TODO ##
        #return the FOV evaluated at s_query
        pass
    
    
        
        
        
        
        