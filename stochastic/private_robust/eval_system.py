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
from generate_Q import *


# -----------------------------------------------------------------------------#

# returns time of success; if no success, return num_time_steps
def get_convergence_time(ratio, min_ratio):
    for t in range(len(ratio)):     
        if ratio[t] <= min_ratio:
            return t
    return t
            
plot_run = False

# -----------------------------------------------------------------------------#
# load data

run = 'RC02'
prefix = "../data/RCx/" + run + "_"

range_alpha = pickle.load(open(prefix+"range_alpha.p", "rb"))
range_beta = pickle.load(open(prefix+"range_beta.p", "rb"))
range_lambda = pickle.load(open(prefix+"range_lambda.p", "rb"))
num_sample_iter = pickle.load(open(prefix+"num_sample_iter.p", "rb"))
num_timesteps = pickle.load(open(prefix+"num_timesteps.p", "rb"))

traj_ratio = pickle.load(open(prefix+"traj_ratio.p", "rb"))



# -----------------------------------------------------------------------------#
# compute results


min_ratio = 0.05

t_min = np.zeros((num_sample_iter))
success = np.zeros((num_sample_iter))

for el in range(len(range_lambda)):
    lap = range_lambda[el]
    for a in range(len(range_alpha)):
        alpha = range_alpha[a]
        for b in range(len(range_beta)):        
            beta = range_beta[b]

            tr = traj_ratio[(lap,alpha,beta)]
            for i in range(num_sample_iter):
                t_min[i] = get_convergence_time(tr[:,i], min_ratio)  
                success[i] = (t_min[i]<num_timesteps-1)    
            print "Success rate: ", sum(success)/len(success)
            fig = plt.figure()
            plt.hist(t_min, bins=50, range=[0, num_timesteps], normed=False, weights=None)
            

# -----------------------------------------------------------------------------#
# plot

if plot_run:
    i = 0
    fig = plot_traits_ratio_time_mac(deploy_robots_euler_it[:,:,:,i], deploy_traits_desired, species_traits, delta_t, match)
    plt.plot([0, t_max_sim],[min_ratio, min_ratio])
     
   