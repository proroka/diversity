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
from matplotlib.font_manager import FontProperties

# -----------------------------------------------------------------------------#
# load data

verbose = False


run = 'RC11'
prefix = "../data/RCx/" + run + "/" + run + "_"

range_alpha = pickle.load(open(prefix+"range_alpha.p", "rb"))
range_beta = pickle.load(open(prefix+"range_beta.p", "rb"))
range_lambda = pickle.load(open(prefix+"range_lambda.p", "rb"))
num_sample_iter = pickle.load(open(prefix+"num_sample_iter.p", "rb"))
num_timesteps = pickle.load(open(prefix+"num_timesteps.p", "rb"))

traj_ratio = pickle.load(open(prefix+"traj_ratio.p", "rb"))


#------------------

def GetColors(n):
    cm = plt.get_cmap('gist_rainbow')
    return [cm(float(i) / float(n)) for i in range(n)]

# -----------------------------------------------------------------------------#
# compute results


min_ratio = 0.02

t_min = np.zeros((num_sample_iter))
success = np.zeros((num_sample_iter))

success_values = {}
t_avg_values = {}
t_std_values = {}
t_med_values = {}

colors = GetColors(len(range_lambda))
fontP = FontProperties()
fontP.set_size('small')

   
fig1 = plt.figure(1,figsize=(5,4))
fig2 = plt.figure(2,figsize=(5,4))
legends = []

lines1 = []
lines2 = []

lab = []
labb = []

for el in range(len(range_lambda)):
    lap = range_lambda[el]

    success_values[el] = np.zeros(len(range_alpha))
    t_avg_values[el] = np.zeros(len(range_alpha))
    t_std_values[el] = np.zeros(len(range_alpha))
    t_med_values[el] = np.zeros(len(range_alpha))

    col = colors[el]    
    
    for a in range(len(range_alpha)):
        b = a
        
        alpha = range_alpha[a]
        beta = range_beta[b]
        tr = traj_ratio[(el, a, b)]
      
        # compute statistics
        for i in range(num_sample_iter):
            t_min[i] = get_convergence_time(tr[:,i], min_ratio)  
            success[i] = (t_min[i]<num_timesteps-1)    
        if verbose:
            print "Success rate: ", sum(success)/len(success)
            
        # store values
        success_values[el][a] = sum(success)/len(success)
        t_sorted = t_min[t_min<num_timesteps-1]
        t_avg_values[el][a] = np.mean(t_sorted)            
        t_med_values[el][a] = np.median(t_sorted)            
        t_std_values[el][a] = np.std(t_sorted)
        
          
    plt.figure(1)
    x = range(len(range_alpha))
    
    lines1.append(plt.plot(x, t_avg_values[el], c=col, lw=2)[0])
    
    plt.errorbar(x, t_avg_values[el],t_std_values[el],lw=2,c=col)
    plt.figure(2)
    lines2.append(plt.plot(x, success_values[el], c=col, lw=2)[0])

    legends.append(('lap = %.2f' % lap))


x_axl = np.arange(0,len(range_alpha),2)    
for a in range(len(range_alpha)):  
    if a in x_axl:
        alpha = range_alpha[a]
        beta = range_beta[a]
        lab.append('%.2f / %.2f' % (alpha, beta))

      
plt.figure(1)        
ax = plt.gca()
ax.set_ylim([0, num_timesteps])   
ax.set_xticks(x_axl)
ax.set_xticklabels(lab)
plt.legend(lines1, legends, prop=fontP, loc=2, borderaxespad=0., bbox_to_anchor=(1.05, 1)) 
plt.title('Time')
s = 'run_' + run + 'time.eps'
fig1.savefig(s)

plt.figure(2)
ax = plt.gca()
ax.set_ylim([0, 1.1])  
ax.set_xticks(x_axl)
ax.set_xticklabels(lab) 
plt.legend(lines2, legends, prop=fontP, loc=2, borderaxespad=0., bbox_to_anchor=(1.05, 1)) 
plt.title('Success')
s = 'run_' + run + 'success.eps'
fig2.savefig(s)
# -----------------------------------------------------------------------------#
# plot trajectories

#