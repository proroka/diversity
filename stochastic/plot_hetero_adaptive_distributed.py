# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:40:29 2015
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
sys.path.append('plotting')
sys.path.append('utilities')
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod
from generate_Q import *
from simple_orrank import *



#load data
run = 'D06'
prefix = "./data/" + run + "/" + run + '_'
save_plots = False

delta_t = 0.04 # time step
match = 1

graph = pickle.load(open(prefix+"graph.p", "rb"))
species_traits = pickle.load(open(prefix+"species_traits.p", "rb"))
deploy_robots_init = pickle.load(open(prefix+"deploy_robots_init.p", "rb"))
deploy_traits_init = pickle.load(open(prefix+"deploy_traits_init.p", "rb"))
deploy_traits_desired = pickle.load(open(prefix+"deploy_traits_desired.p", "rb"))
deploy_robots_micro_adapt = pickle.load(open(prefix+"deploy_robots_micro_adapt.p", "rb"))
deploy_robots_micro = pickle.load(open(prefix+"deploy_robots_micro.p", "rb"))
deploy_robots_euler = pickle.load(open(prefix+"deploy_robots_euler.p", "rb"))
deploy_robots_micro_adapt_hop = pickle.load(open(prefix+"deploy_robots_micro_adapt_hop.p", "rb"))

# cut off end:
fin = 170
deploy_robots_micro_adapt_hop = deploy_robots_micro_adapt_hop[:,0:fin,:,:,:]
deploy_robots_euler = deploy_robots_euler[:,0:fin,:]

fig = plot_traits_ratio_time_mic_distributed(deploy_robots_micro_adapt_hop,deploy_robots_euler, deploy_traits_desired, species_traits, delta_t, match)

#plt.axes().set_aspect(0.65,'box')
plt.show()

# get time of convergence to min-ratio
min_ratio = 0.07
fig = plt.figure()
ax = plt.gca()

num_hops = deploy_robots_micro_adapt_hop.shape[4]
num_graph_iter = deploy_robots_micro_adapt_hop.shape[3]
t_min_d = np.zeros((num_hops, num_graph_iter))
for it in range(num_graph_iter):
    for nh in range(num_hops):
        t_min_d[nh,it] = get_traits_ratio_time(deploy_robots_micro_adapt_hop[:,:,:,it,nh], deploy_traits_desired, species_traits, match, min_ratio) 

  
x = range(1,num_hops+1)
tmin = np.transpose(t_min_d) * delta_t
bp = plt.boxplot(tmin, positions=x, widths=0.2, notch=0, sym='+', vert=1, whis=1.5) 
    
ymin = 0 
ymax = 7
ax.set_ylim([0, ymax])    
#ax.set_xlim([1, num_hops])

plt.show()