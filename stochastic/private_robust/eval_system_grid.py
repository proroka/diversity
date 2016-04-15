# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 19:06:26 2016
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
plot_traj = False
pareto_scatter = True

run = 'RC14'
prefix = "../data/RCx/" + run + "_"

range_alpha = pickle.load(open(prefix+"range_alpha.p", "rb"))
range_beta = pickle.load(open(prefix+"range_beta.p", "rb"))
range_lambda = pickle.load(open(prefix+"range_lambda.p", "rb"))
num_sample_iter = pickle.load(open(prefix+"num_sample_iter.p", "rb"))
num_timesteps = pickle.load(open(prefix+"num_timesteps.p", "rb"))

traj_ratio = pickle.load(open(prefix+"traj_ratio.p", "rb"))


def GetColors(n):
    cm = plt.get_cmap('gist_rainbow')
    return [cm(float(i) / float(n)) for i in range(n)]
    

# -----------------------------------------------------------------------------#
# compute results


min_ratio = 0.015

t_min = np.zeros((num_sample_iter))
success = np.zeros((num_sample_iter))

success_values = {}
t_avg_values = {}
t_std_values = {}
t_med_values = {}

for el in range(len(range_lambda)):
    lap = range_lambda[el]
    success_values[lap] = np.zeros((len(range_alpha), len(range_beta)))
    t_avg_values[lap] = np.zeros((len(range_alpha), len(range_beta)))
    t_std_values[lap] = np.zeros((len(range_alpha), len(range_beta)))
    t_med_values[lap] = np.zeros((len(range_alpha), len(range_beta)))
    
    pareto_time = []
    pareto_succ = []
    pareto_cols_a = []
    pareto_cols_b = []
    for a in range(len(range_alpha)):
        alpha = range_alpha[a]
        for b in range(len(range_beta)):  
            pareto_cols_a.append(colors[a])
            pareto_cols_b.append(colors[b])
            
            beta = range_beta[b]
            # get data
            tr = traj_ratio[(el,a,b)]
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
            t_med_values[lap][a,b] = np.median(t_sorted)            
            t_std_values[lap][a,b] = np.std(t_sorted)
            
            # get data for scatter plot
            pareto_time.append(t_avg_values[lap][a,b])
            pareto_succ.append(success_values[lap][a,b])
            
            if(plot_hist):
                fig = plt.figure()
                plt.hist(t_min, bins=50, range=[0, num_timesteps], normed=False, weights=None)
    
    # plot values for each Laplace noise value

    if plot_grid:
        cmap = plt.get_cmap('Reds')  
        
        fig1 = plt.figure(figsize=(4,4))
        vals_nonnan = success_values[lap][~np.isnan(success_values[lap])]
        clim = (np.min(vals_nonnan), np.max(vals_nonnan))
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
        s = 'run_' + run + '_lap_' + str(lap) + '_success.eps'
        fig.savefig(s)
    
        fig2 = plt.figure(figsize=(4,4))
        vals_nonnan = t_avg_values[lap][~np.isnan(t_avg_values[lap])]
        clim = (np.min(vals_nonnan), np.max(vals_nonnan))
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
        s = 'run_' + run + '_lap_' + str(lap) + '_tconv_mean.eps'
        fig2.savefig(s)
    

        
    if pareto_scatter:
        
        
        fig3 = plt.figure(figsize=(4,4))
        plt.scatter(pareto_time, pareto_succ, s=15, c=pareto_cols_a)
        plt.xlim([0, num_timesteps])
        plt.ylim([0, 1.])
        s = 'run_' + run + '_lap_' + str(lap) + '_pareto_a.eps'
        fig3.savefig(s)
        
        fig3 = plt.figure(figsize=(4,4))
        plt.scatter(pareto_time, pareto_succ, s=15, c=pareto_cols_b)
        plt.xlim([0, num_timesteps])
        plt.ylim([0, 1.])
        s = 'run_' + run + '_lap_' + str(lap) + '_pareto_b.eps'
        fig3.savefig(s)