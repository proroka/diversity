o# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 09:07:09 2015

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
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod


#load data
run = 'V05'
prefix = "./data/" + run + "/"
save_plots = False

graph = pickle.load(open(prefix+"graph.p", "rb"))
species_traits = pickle.load(open(prefix+"species_traits.p", "rb"))
deploy_robots_init = pickle.load(open(prefix+"deploy_robots_init.p", "rb"))
deploy_traits_init = pickle.load(open(prefix+"deploy_traits_init.p", "rb"))
deploy_traits_desired = pickle.load(open(prefix+"deploy_traits_desired.p", "rb"))
deploy_robots_micro_adapt = pickle.load(open(prefix+"deploy_robots_micro_adapt.p", "rb"))
deploy_robots_micro = pickle.load(open(prefix+"deploy_robots_micro.p", "rb"))
deploy_robots_euler = pickle.load(open(prefix+"deploy_robots_euler.p", "rb"))

num_nodes = deploy_robots_init.shape[0]
num_traits = species_traits.shape[1]
num_species = species_traits.shape[0]
o
# get averages
avg_deploy_robots_micro_adapt = np.mean(deploy_robots_micro_adapt,3)
avg_deploy_robots_micro = np.mean(deploy_robots_micro,3)

# plot
species_ind = 0
node_ind = [4, 5]
fig1 = plot_robots_time_micmac(avg_deploy_robots_micro, deploy_robots_euler, species_ind, node_ind)
#plt.axis('square')
plt.axes().set_aspect(8/200.0,'box')
plt.show()

trait_ind = 3
fig2 = plot_traits_time_micmac(avg_deploy_robots_micro, deploy_robots_euler, species_traits, node_ind, trait_ind)
plt.axes().set_aspect(8/350.0,'box')
plt.legend(loc='center right')
plt.show()

delta_t = 0.04
match = 1
fig3 = plot_traits_ratio_time_micmicmac(deploy_robots_micro, deploy_robots_micro_adapt, 
                                        deploy_robots_euler, deploy_traits_desired, 
                                        species_traits, delta_t, match)
plt.show()

fig6 = plot_traits_ratio_time_micmac(deploy_robots_micro, deploy_robots_euler, 
                                     deploy_traits_desired, species_traits, delta_t, match)
plt.show()

# plot graph
plt.axis('equal')
fig4 = nxmod.draw_circular(deploy_traits_init, graph,linewidths=3)
plt.show()
plt.axis('equal')
fig5  = nxmod.draw_circular(deploy_traits_desired, graph, linewidths=3)
plt.show()
adjacency_m = nx.to_numpy_matrix(graph)
adjacency_m = np.squeeze(np.asarray(adjacency_m))


if save_plots:
    fig1.savefig('./plots/'+run+'/num_robots_time.eps')              
    fig2.savefig('./plots/'+run+'/num_traits_time.eps')      
    fig3.savefig('./plots/'+run+'/ratio_time_micmicmac.eps')              
    fig4.savefig('./plots/'+run+'/graph_init.eps') 
    fig5.savefig('./plots/'+run+'/graph_desired.eps') 
    fig6.savefig('./plots/'+run+'/ratio_time_micmac.eps') 






