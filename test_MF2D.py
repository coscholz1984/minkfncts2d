# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 20:25:09 2021

@author: cnceo
"""

import matplotlib.pyplot as plt
import numpy as np
from MF2D import MF2Dfunc # This is the essential function to calculate MFs

# this function generates a test-pattern (image) of gaussian noise 
# and calculates the MF in the range given by thresholds
def test_gaussian(shape, thresholds):
    # inputs: shape = 2-tupel, size of the image (x,y)
    #         thresholds = linspace, e.g. np.linspace(0.0, 1.0, 100)
    image = np.clip(np.random.normal(0.5, 0.1, shape), 0.0, 1.0)
    
    # Here we calculate the 3 functions, F, U and Chi
    (F, U, Chi) = MF2Dfunc(image, thresholds)

    return (F, U, Chi, image)

# Show an example of MFs of Gaussian white noise
thresholds = np.linspace(0.0, 1.0, 100)
(F, U, Chi, image) = test_gaussian((128, 128), thresholds)

plt.figure(1)
plt.clf()
plt.title("Image with Gaussian Noise")
plt.imshow(image, cmap="viridis")
plt.show()

plt.figure(1)
plt.clf()
plt.plot(thresholds, F, thresholds, U, thresholds, Chi)
plt.title("2D Minkowski Functions")
plt.legend(["F (Area)", "U (Boundary)", "$\chi$ (Euler characteristic)"])
plt.xlabel("Threshold")
plt.show()