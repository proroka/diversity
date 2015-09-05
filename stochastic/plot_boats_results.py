# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:13:22 2015
@author: amandaprorok

"""

import numpy as np
import scipy as sp
import scipy.io
import scipy.ndimage.filters
import pylab as pl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import sys
import pickle
import time


def plot_traits_ratio_time(ax, deploy_robots_1, deploy_robots_2, deploy_traits_desired, transform, delta_t, match, color, label):
    num_tsteps = deploy_robots_1.shape[1]
    total_num_traits = np.sum(deploy_traits_desired)
    diffmac_rat_1 = np.zeros(num_tsteps)
    diffmac_rat_2 = np.zeros(num_tsteps)

    for t in range(num_tsteps):
        if match==0:
            traits = np.dot(deploy_robots_1[:,t,:], transform)
            diffmac = np.abs(np.minimum(traits - deploy_traits_desired, 0))
        else:
            traits = np.dot(deploy_robots_1[:,t,:], transform)
            diffmac = np.abs(traits - deploy_traits_desired)
        diffmac_rat_1[t] = np.sum(diffmac) / total_num_traits
    for t in range(num_tsteps):
        if match==0:
            traits = np.dot(deploy_robots_2[:,t,:], transform)
            diffmac = np.abs(np.minimum(traits - deploy_traits_desired, 0))
        else:
            traits = np.dot(deploy_robots_2[:,t,:], transform)
            diffmac = np.abs(traits - deploy_traits_desired)
        diffmac_rat_2[t] = np.sum(diffmac) / total_num_traits
    
    # plot average of 2 runs    
    #diffmac_rat = (diffmac_rat_1 + diffmac_rat_2) / 2.0   
    diffmac_rat_all = np.vstack((np.asarray(diffmac_rat_1), np.asarray(diffmac_rat_2)));    
    diffmac_rat_avg = np.mean(diffmac_rat_all, axis=0) / 2.0; # remove double-counting of misplaced traits
    diffmac_rat_std = np.std(diffmac_rat_all, axis=0)
    
    x = np.squeeze(np.arange(0, num_tsteps) * delta_t)
    

    l2 = ax.plot(x, diffmac_rat_avg, color=color, linewidth=2, label=label)

    err_ax = range(1,len(x),100)
    #ax.errorbar(x[err_ax],diffmac_rat_avg[err_ax],diffmac_rat_std[err_ax],linestyle='.',color='black',linewidth=2)
    plt.xlim([0,400])
    
    return fig
    
        
##-------------------------------
# Two experiment sets: rank-1 and rank-2
# rank 1: 23, 26
# rank 2: 25, 22
##-------------------------------

save_plots = True

rk1 = False

run = 25 # choose run
prefix = 'data/boats/run_' + str(run) + '_'
deploy_boats_1 = pickle.load(open(prefix+"deploy_boats.p", "rb"))

run = 22 # choose run
prefix = 'data/boats/run_' + str(run) + '_'
deploy_boats_2 = pickle.load(open(prefix+"deploy_boats.p", "rb"))

# load rest of data
delta_t = pickle.load(open("delta_t", "rb"))
t_max = pickle.load(open("t_max.p", "rb"))
match = pickle.load(open("match.p", "rb"))
K = pickle.load(open(prefix+"K.p", "rb"))
deploy_robots_init = pickle.load(open(prefix+"deploy_robots_init.p", "rb"))
deploy_robots_euler = pickle.load(open(prefix+"deploy_robots_euler.p", "rb"))
deploy_traits_desired = pickle.load(open(prefix+"deploy_traits_desired.p", "rb"))
species_traits = pickle.load(open(prefix+"species_traits.p", "rb"))





fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=True)

plot_traits_ratio_time(ax, deploy_robots_euler, deploy_robots_euler, deploy_traits_desired, species_traits, delta_t, match, 'blue', 'Macroscopic')
plot_traits_ratio_time(ax, deploy_boats_1, deploy_boats_2, deploy_traits_desired, species_traits, delta_t, match, 'green', 'Boats')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Ratio of misplaced traits')
ax.set_aspect(400/1,'box') 
plt.grid(axis='y')
plt.legend()
plt.show()

if save_plots:
    if rk1:
        fig.savefig('results_boats_rk1.eps') 
    else:
        fig.savefig('results_boats_rk2.eps') 

