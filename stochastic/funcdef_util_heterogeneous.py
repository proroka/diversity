# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:26:07 2015
@author: amanda

"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx


# -----------------------------------------------------------------------------#
# plot ratio of desired vs actual robot distribution

def plot_robots_ratio_time(deploy_robots, deploy_robots_desired):
    
    num_iter = deploy_robots.shape[1]

    diffsqs = np.zeros(num_iter)
    for t in range(num_iter):
        diff = deploy_robots[:,t,:] - deploy_robots_desired
        diffsqs[t] = np.sum(np.square(diff))

    x = np.arange(0, num_iter)
    plt.plot(x,diffsqs)
    plt.show()
    
    
# -----------------------------------------------------------------------------#
# initialize robots
    
def plot_traits_ratio_time(deploy_robots, deploy_traits_desired, transform):
    
    num_iter = deploy_robots.shape[1]

    diffsqs = np.zeros(num_iter)
    for t in range(num_iter):
        traits = np.dot(deploy_robots[:,t,:], transform)
        diff = traits - deploy_traits_desired
        diffsqs[t] = np.sum(np.square(diff))

    x = np.arange(0, num_iter)
    plt.plot(x,diffsqs)
    plt.show()