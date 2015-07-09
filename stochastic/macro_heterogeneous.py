# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:53:24 2015
@author: amandaprorok

"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx

# my modules
from optimize_transition_matrix_hetero_fast_gradient import *
from funcdef_macro_heterogeneous import *
from funcdef_util_heterogeneous import *

# -----------------------------------------------------------------------------#

# 1. create graph
# 2. initialize robot deployment on nodes
# 3. initialize transition matrix
# 4. run macro-discrete model 
# 5. plot evolution of robot population per node on graph


# -----------------------------------------------------------------------------#
# initialize world and robot community

# general
t_max = 10.0 # time in seconds
rate_max = 5.0 # maximum transition rate

# create network of sites
size_lattice = 3
num_nodes = size_lattice**2

# set of traits
num_traits = 8
max_trait_values = 2 # [0,1]: trait availability

# robot species
num_species = 3
max_robots = 10 # maximum number of robots per node
deploy_robots_init = np.random.randint(0, max_robots, size=(num_nodes, num_species))

species_traits = np.random.randint(0, max_trait_values, (num_species, num_traits))
deploy_traits_init = np.dot(deploy_robots_init, species_traits)

# random end state
random_transition = random_transition_matrix(num_nodes, rate_max/2)

# sample final desired trait distribution based on random transition matrix
deploy_robots_final = sample_final_robot_distribution(deploy_robots_init, random_transition, t_max)
deploy_traits_desired = np.dot(deploy_robots_final, species_traits)

# -----------------------------------------------------------------------------#


print "total robots, init: \t", np.sum(np.sum(deploy_robots_init))
print "total robots, final: \t", np.sum(np.sum(deploy_robots_final))
print "total traits, init: \t", np.sum(np.sum(deploy_traits_init))
print "total traits, desired:\t", np.sum(np.sum(deploy_traits_desired))


# -----------------------------------------------------------------------------#
# initialize graph: all species move on same graph

graph = nx.grid_2d_graph(size_lattice, size_lattice)
#graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.5) #, tries=100, seed=None)
# get the adjencency matrix
adjacency_m = nx.to_numpy_matrix(graph)
adjacency_m = np.squeeze(np.asarray(adjacency_m))


# -----------------------------------------------------------------------------#
# find optimal transition matrix

transition_m = optimal_transition_matrix(adjacency_m, deploy_robots_init, deploy_traits_desired, species_traits, t_max, rate_max)

      
# -----------------------------------------------------------------------------#
# run euler integration to drive robots to end state

deploy_robots = run_euler_integration(deploy_robots_init, transition_m, t_max)

deploy_robots_final = deploy_robots[:,-1,:]
deploy_traits_final = np.dot(deploy_robots_final, species_traits) 


# -----------------------------------------------------------------------------#
# plots

#plot_network(graph, deploy_traits_init, deploy_traits_final)
# plot_histogram(deploy_traits_final)

# plot evolution over time
species_index = 0
trait_index = 0
plot_robots_time(deploy_robots, species_index)
#plot_traits_time(deploy_robots, species_traits, trait_index)
 
   
plot_robots_ratio_time(deploy_robots, deploy_robots_final)  
plot_traits_ratio_time(deploy_robots, deploy_traits_desired, species_traits)






