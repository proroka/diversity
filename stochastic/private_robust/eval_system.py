# -*- coding: utf-8 -*-
"""
Created on Tues April  4 18:08:19 2016
@author: amandaprorok

"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import pickle

# my modules
sys.path.append('../plotting')
sys.path.append('../utilities')
sys.path.append('..')
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod
from scipy import interpolate

# -----------------------------------------------------------------------------#
# utilities

# returns time of success; if no success, return num_time_steps
def get_convergence_time(ratio, min_ratio):
    for t in range(len(ratio)):     
        if ratio[t] <= min_ratio:
            return t
    return t
            

# -----------------------------------------------------------------------------#
# load data

plot_run = False
t_hist_plots = False
verbose = False

run = 'RC06'
prefix = "../data/RCx/" + run + "_"

range_alpha = pickle.load(open(prefix+"range_alpha.p", "rb"))
range_beta = pickle.load(open(prefix+"range_beta.p", "rb"))
range_lambda = pickle.load(open(prefix+"range_lambda.p", "rb"))
num_sample_iter = pickle.load(open(prefix+"num_sample_iter.p", "rb"))
num_timesteps = pickle.load(open(prefix+"num_timesteps.p", "rb"))

traj_ratio = pickle.load(open(prefix+"traj_ratio.p", "rb"))



# -----------------------------------------------------------------------------#
# compute results


min_ratio = 0.02

t_min = np.zeros((num_sample_iter))
success = np.zeros((num_sample_iter))

success_values = {}
t_avg_values = {}
t_std_values = {}

for el in range(len(range_lambda)):
    lap = range_lambda[el]
    success_values[lap] = np.zeros((len(range_alpha), len(range_beta)))
    t_avg_values[lap] = np.zeros((len(range_alpha), len(range_beta)))
    t_std_values[lap] = np.zeros((len(range_alpha), len(range_beta)))
    
    for a in range(len(range_alpha)):
        alpha = range_alpha[a]
        for b in range(len(range_beta)):        
            beta = range_beta[b]

            tr = traj_ratio[(lap,alpha,beta)]
            for i in range(num_sample_iter):
                t_min[i] = get_convergence_time(tr[:,i], min_ratio)  
                success[i] = (t_min[i]<num_timesteps-1)    
            if verbose:
                print "Success rate: ", sum(success)/len(success)

            # store values
            success_values[lap][a,b] = sum(success)/len(success)
            t_sorted = t_min[t_min<num_timesteps-1]
            t_avg_values[lap][a,b] = np.mean(t_sorted)            
            t_std_values[lap][a,b] = np.std(t_sorted)
            
            if(t_hist_plots):
                fig = plt.figure()
                plt.hist(t_min, bins=50, range=[0, num_timesteps], normed=False, weights=None)
    
    # plot values for each Laplace noise value

    cmap = plt.get_cmap('Reds')  
    extent=[range_alpha[0], range_alpha[-1], range_beta[0], range_beta[-1]]
    
    plt.imshow(success_values[lap], interpolation='nearest', origin='lower', cmap=cmap)
    ax = plt.axes()
    plt.colorbar()
    plt.title('Success Rates')
    plt.xlabel('beta')
    plt.xticks(range(len(range_beta)))
    ax.set_xticklabels(range_beta)
    plt.ylabel('alpha')
    plt.yticks(range(len(range_alpha)))
    ax.set_yticklabels(range_alpha)
    plt.show()

    plt.imshow(t_avg_values[lap], interpolation='nearest', origin='lower', cmap=cmap)
    ax = plt.axes()
    plt.colorbar()
    plt.title('Mean Convergence Time')
    plt.xlabel('beta')
    plt.xticks(range(len(range_beta)))
    ax.set_xticklabels(range_beta)
    plt.ylabel('alpha')
    plt.yticks(range(len(range_alpha)))
    ax.set_yticklabels(range_alpha)
    plt.show()        

    plt.imshow(t_std_values[lap], interpolation='nearest', origin='lower', cmap=cmap)
    ax = plt.axes()    
    plt.colorbar()
    plt.title('Std. Convergence Time')
    plt.xlabel('beta')
    plt.xticks(range(len(range_beta)))
    ax.set_xticklabels(range_beta)
    plt.ylabel('alpha')
    plt.yticks(range(len(range_alpha)))
    ax.set_yticklabels(range_alpha)
    plt.show() 
    
# -----------------------------------------------------------------------------#
# plot

if plot_run:
    i = 0
    fig = plot_traits_ratio_time_mac(deploy_robots_euler_it[:,:,:,i], deploy_traits_desired, species_traits, delta_t, match)
    plt.plot([0, t_max_sim],[min_ratio, min_ratio])
     
   