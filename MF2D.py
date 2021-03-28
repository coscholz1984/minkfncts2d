# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:57:41 2021

@author: Christian Scholz
Fork of https://github.com/moutazhaq/minkfncts2d

# Implementation based on http://iopscience.iop.org/article/10.1088/1742-5468/2008/12/P12015
# and https://github.com/moutazhaq/minkfncts2d
"""
import numpy as np
import math
from numba import jit

@jit(nopython=True, parallel=False, error_model="numpy")
def MF2Dfunc(image, thresholds):
    # np.ndarray[np.double_t, ndim=2] image, np.linspace(0.0, 1.0, 100) thresholds
    F = []
    U = []
    Chi = []
    
    for threshold in thresholds:
        (f, u, chi) = MF2D(image, threshold)
        F.append(f)
        U.append(u)
        Chi.append(chi)

    return (F, U, Chi)

@jit(nopython=True, parallel=False, error_model="numpy")
def MF2D(image, threshold):
    # np.ndarray[np.double_t, ndim=2] image, double threshold
    height = image.shape[0]
    width = image.shape[1]
    
    f = 0.0
    u = 0.0
    chi = 0.0

    for y in range(height-1):
        p10 = image[y, 0]
        p11 = image[y+1, 0]
        for x in range(width-1):
            pattern = 0
            
            p00 = p10
            p01 = p11
            p10 = image[y, x+1]
            p11 = image[y+1, x+1]
            
            if p00 > threshold:
                pattern = pattern | 1
            if p10 > threshold:
                pattern = pattern | 2
            if p11 > threshold:
                pattern = pattern | 4
            if p01 > threshold:
                pattern = pattern | 8
                
            # a1 = (p00 - threshold) / (p00 - p10)
            # a2 = (p10 - threshold) / (p10 - p11)
            # a3 = (p01 - threshold) / (p01 - p11)
            # a4 = (p00 - threshold) / (p00 - p01)
            
            if pattern == 0:
                pass
            elif pattern == 1:
                a1 = (p00 - threshold) / (p00 - p10)
                a4 = (p00 - threshold) / (p00 - p01)
                f += 0.5 * a1 * a4
                u += math.sqrt(a1*a1 + a4*a4)
                chi += 0.25
            elif pattern == 2:
                a1 = (p00 - threshold) / (p00 - p10)
                a2 = (p10 - threshold) / (p10 - p11)
                f += 0.5 * (1.0-a1)*a2
                u += math.sqrt((1.0-a1)*(1.0-a1) + a2*a2)
                chi += 0.25
            elif pattern == 3:
                a2 = (p10 - threshold) / (p10 - p11)
                a4 = (p00 - threshold) / (p00 - p01)
                f += a2 + 0.5*(a4-a2)
                u += math.sqrt(1.0 + (a4-a2)*(a4-a2))
            elif pattern == 4:
                a2 = (p10 - threshold) / (p10 - p11)
                a3 = (p01 - threshold) / (p01 - p11)
                f += 0.5 * (1.0-a2)*(1.0-a3)
                u += math.sqrt((1.0-a2)*(1.0-a2) + (1.0-a3)*(1.0-a3))
                chi += 0.25
            elif pattern == 5:
                a1 = (p00 - threshold) / (p00 - p10)
                a2 = (p10 - threshold) / (p10 - p11)
                a3 = (p01 - threshold) / (p01 - p11)
                a4 = (p00 - threshold) / (p00 - p01)
                f += 1.0 - 0.5*(1.0-a1)*a2 - 0.5*a3*(1.0-a4)
                u += math.sqrt((1.0-a1)*(1.0-a1) + a2*a2) + math.sqrt(a3*a3 + (1.0-a4)*(1.0-a4))
                chi += 0.5
            elif pattern == 6:
                a1 = (p00 - threshold) / (p00 - p10)
                a3 = (p01 - threshold) / (p01 - p11)
                f += (1.0-a3) + 0.5*(a3-a1)
                u += math.sqrt(1.0 + (a3-a1)*(a3-a1))
            elif pattern == 7:
                a3 = (p01 - threshold) / (p01 - p11)
                a4 = (p00 - threshold) / (p00 - p01)
                f += 1.0 - 0.5*a3*(1.0-a4)
                u += math.sqrt(a3*a3 + (1.0-a4)*(1.0-a4))
                chi += -0.25
            elif pattern == 8:
                a3 = (p01 - threshold) / (p01 - p11)
                a4 = (p00 - threshold) / (p00 - p01)
                f += 0.5*a3*(1.0-a4)
                u += math.sqrt(a3*a3 + (1.0-a4)*(1.0-a4))
                chi += 0.25
            elif pattern == 9:
                a1 = (p00 - threshold) / (p00 - p10)
                a3 = (p01 - threshold) / (p01 - p11)
                f += a1 + 0.5*(a3-a1)
                u += math.sqrt(1.0 + (a3-a1)*(a3-a1))
            elif pattern == 10:
                a1 = (p00 - threshold) / (p00 - p10)
                a2 = (p10 - threshold) / (p10 - p11)
                a3 = (p01 - threshold) / (p01 - p11)
                a4 = (p00 - threshold) / (p00 - p01)
                f += 1.0 - 0.5*a1*a4 + 0.5*(1.0-a2)*(1.0-a3)
                u += math.sqrt(a1*a1 + a4*a4) + math.sqrt((1.0-a2)*(1.0-a2) + (1.0-a3)*(1.0-a3))
                chi += 0.5
            elif pattern == 11:
                a2 = (p10 - threshold) / (p10 - p11)
                a3 = (p01 - threshold) / (p01 - p11)
                f += 1.0 - 0.5*(1.0-a2)*(1.0-a3)
                u += math.sqrt((1.0-a2)*(1.0-a2) + (1.0-a3)*(1.0-a3))
                chi += -0.25
            elif pattern == 12:
                a2 = (p10 - threshold) / (p10 - p11)
                a4 = (p00 - threshold) / (p00 - p01)
                f += (1.0-a2) + 0.5*(a2-a4)
                u += math.sqrt(1.0 + (a2-a4)*(a2-a4))
            elif pattern == 13:
                a1 = (p00 - threshold) / (p00 - p10)
                a2 = (p10 - threshold) / (p10 - p11)
                f += 1.0 - 0.5*(1.0-a1)*a2
                u += math.sqrt((1.0-a1)*(1.0-a1) + a2*a2)
                chi += -0.25
            elif pattern == 14:
                a1 = (p00 - threshold) / (p00 - p10)
                a4 = (p00 - threshold) / (p00 - p01)
                f += 1.0 - 0.5*a1*a4
                u += math.sqrt(a1*a1 + a4*a4)
                chi += -0.25
            elif pattern == 15:
                f += 1.0
    return (f, u, chi)