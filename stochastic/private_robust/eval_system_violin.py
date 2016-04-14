# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:28:15 2016
@author: amanda

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

   
fig1 = plt.figure(1,figsize=(8,4))
fig2 = plt.figure(2,figsize=(8,4))
legends = []

lines1 = []
lines2 = []

lab = []
labb = []

# choose a laplace noise value for violin plot
el = 2


lap = range_lambda[el]

success_values[el] = np.zeros(len(range_alpha))
t_avg_values[el] = np.zeros(len(range_alpha))
t_std_values[el] = np.zeros(len(range_alpha))
t_med_values[el] = np.zeros(len(range_alpha))

t_sorted_values = np.zeros((len(range_alpha), num_sample_iter))

col = colors[el]    
data = []

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
    temp_len = len(t_min[t_min<num_timesteps-1])
    
    temp = t_min[t_min<num_timesteps-1]
 
    t_avg_values[el][a] = np.mean(t_sorted)            
    t_med_values[el][a] = np.median(t_sorted)            
    t_std_values[el][a] = np.std(t_sorted)

    data.append(temp)
      
plt.figure(1)
x = range(len(range_alpha))
plt.violinplot(data, x, showmedians=True)

for a in range(len(range_alpha)):  
    alpha = range_alpha[a]
    beta = range_beta[a]    
    lab.append('%.2f / %.2f' % (alpha, beta))

plt.title('lap %.2f' % lap)
fig1.savefig('violin_lap = %.2f.eps' % lap)
    
ax = plt.gca()
ax.set_xticks(range(len(range_alpha)))
ax.set_xticklabels(lab) 

