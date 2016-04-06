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
from funcdef_util_privacy import *


# -----------------------------------------------------------------------------#
# load data

plot_run = False
plot_hist = False
plot_grid = True
verbose = False
selected_runs = True

run = 'RC09'
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
            # get data
            tr = traj_ratio[(lap,alpha,beta)]
            # compute statistics
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
            
            if(plot_hist):
                fig = plt.figure()
                plt.hist(t_min, bins=50, range=[0, num_timesteps], normed=False, weights=None)
    
    # plot values for each Laplace noise value

    if plot_grid and not selected_runs:
        cmap = plt.get_cmap('Reds')  
        
        fig = plt.figure(figsize=(4,4))
        clim = (np.min(success_values[lap]), np.max(success_values[lap]))
        plt.imshow(success_values[lap], interpolation='nearest', origin='lower', cmap=cmap, clim=clim)
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
        s = 'lap_' + str(lap) + '_success.eps'
        fig.savefig(s)
    
        fig = plt.figure(figsize=(4,4))
        clim = (np.min(t_avg_values[lap]), np.max(t_avg_values[lap]))
        plt.imshow(t_avg_values[lap], interpolation='nearest', origin='lower', cmap=cmap, clim=clim)
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
        s = 'lap_' + str(lap) + '_tconv_mean.eps'
        fig.savefig(s)
    
        fig = plt.figure(figsize=(4,4))
        clim = (np.min(t_std_values[lap]), np.max(t_std_values[lap]))
        plt.imshow(t_std_values[lap], interpolation='nearest', origin='lower', cmap=cmap, clim=clim)
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
        s = 'lap_' + str(lap) + '_tconv_std.eps'
        fig.savefig(s)
        
    if selected_runs:
        sv = np.flipud(success_values[lap]).diagonal(0)[::-1]        
        tmv = np.flipud(t_avg_values[lap]).diagonal(0)[::-1]
        tsv = np.flipud(t_std_values[lap]).diagonal(0)[::-1]
        fig = plt.figure(figsize=(4,4))
        plt.plot(sv)   
        plt.title('Success')
        plt.xlabel('a incr, b decr')
        fig = plt.figure(figsize=(4,4))
        plt.plot(tmv)  
        plt.xlabel('a incr, b decr')
        plt.title('Mean Time')
        fig = plt.figure(figsize=(4,4))
        plt.plot(tsv)  
        plt.xlabel('a incr, b decr')
        plt.title('Stddev. Time')
        
# -----------------------------------------------------------------------------#
# plot trajectories

# chose relationship between alpha and beta: inverse diag.
# m = (range_alpha[-1] - range_alpha[0]) / (range_beta[-1] - range_beta[0]) * (-1)
# q = range_alpha[-1]
# f_alpha = m * beta + q

for el in range(len(range_lambda)):
    lap = range_lambda[el]
    print 'laplace = ', lap
    # vary b in full range
    fig = plt.figure(figsize=(28,4))    
    for b in range(len(range_beta)):
        beta  = range_beta[b]
        # condition:
        a = relation_ab(b, range_alpha)        
        alpha = range_alpha[a]
        
        # get trajectories
        tr = traj_ratio[(lap,alpha,beta)]    
        tr_mean = np.mean(tr, axis=1)
        tr_std = np.std(tr, axis=1)
        
        # plot ratios
        ax = fig.add_subplot(1,len(range_beta),b+1)
        s = 'a=' + str(alpha) + ' b=' + str(beta)
        plt.title(s)
        ax.set_xticks([0, num_timesteps-1])
        tr_ax = np.arange(0,num_timesteps)
        ind = np.arange(0,num_timesteps,20)       
        plt.plot(tr_mean)
        plt.errorbar(tr_ax[ind], tr_mean[ind], tr_std[ind])
        #for i in range(num_sample_iter):
        #    plt.plot(tr[:,i])
          
    fig.tight_layout()
    plt.show()
    s = 'trajectories_' + str(lap) + '.eps'
    fig.savefig(s)
    
# -----------------------------------------------------------------------------#
# plot

if plot_run:
    i = 0
    fig = plot_traits_ratio_time_mac(deploy_robots_euler_it[:,:,:,i], deploy_traits_desired, species_traits, delta_t, match)
    plt.plot([0, t_max_sim],[min_ratio, min_ratio])
     
   