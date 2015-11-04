# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:14:59 2015
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
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod

# -----------------------------------------------------------------------------#
# initialize world and robot community

t_max = 10.0
delta_t = 0.04
max_rate = 2.0

num_nodes = 10 
# set of traits
num_traits = 4
max_trait_values = 2 # [0,1]: trait availability
# robot species
num_species = 3
max_robots = 300 # maximum number of robots per node
deploy_robots_init = np.random.randint(0, max_robots, size=(num_nodes, num_species))

# change initial robot distribution
deploy_robots_init[1:,:] = 0

# generate a random end state
random_transition = random_transition_matrix(num_nodes, max_rate/2)  # Divide max_rate by 2 for the random matrix to give some slack.        
# sample final desired trait distribution based on random transition matrix   
deploy_robots_final = sample_final_robot_distribution(deploy_robots_init, random_transition, t_max*4., delta_t)
    
deploy_traits_init = np.dot(deploy_robots_init, species_traits)
deploy_traits_desired = np.dot(deploy_robots_final, species_traits)


graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.6)

# plot graph
plt.axis('equal')
fig1 = nxmod.draw_circular(deploy_traits_init, graph,linewidths=3)
plt.show()
plt.axis('equal')
fig2  = nxmod.draw_circular(deploy_traits_desired, graph, linewidths=3)
plt.show()

fig1.savefig('./plots/graph_initial.eps')  
fig2.savefig('./plots/graph_final.eps')  

fig1.savefig('./plots/graph_initial.png')  
fig2.savefig('./plots/graph_final.png')



