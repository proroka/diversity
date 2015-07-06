# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:40:07 2015
@author: amandaprorok

"""


import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx

# my modules
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *

# -----------------------------------------------------------------------------#

# 1. create graph
# 2. initialize robot deployment on nodes
# 3. initialize transition matrix
# 4. run macro-discrete model  / micro simulator
# 5. plot evolution of robot population per node 

# -----------------------------------------------------------------------------#
# initialize world and robot community

# create network of sites
size_lattice = 2
num_nodes = size_lattice**2

# set of traits
num_traits = 2
max_trait_values = 2 # [0,1]: trait availability

# robot species
num_species = 2
max_robots = 10 # maximum number of robots per node
deploy_robots_init = np.random.randint(0, max_robots, size=(num_nodes, num_species))

species_traits = np.random.randint(0, max_trait_values, (num_species, num_traits))
deploy_traits_init = np.dot(deploy_robots_init, species_traits)

# random end state
random_transition = random_transition_matrix(num_nodes)

# sample final desired trait distribution based on random transition matrix
deploy_robots_final = sample_final_robot_distribution(deploy_robots_init, random_transition)
deploy_traits_desired = np.dot(deploy_robots_final, species_traits)

# initialize robots
robots = initialize_robots(deploy_robots_init)
     
    
# -----------------------------------------------------------------------------#

print "total robots, init: \t", np.sum(np.sum(deploy_robots_init))
print "total robots, final: \t", np.sum(np.sum(deploy_robots_final))
print "total traits, init: \t", np.sum(np.sum(deploy_traits_init))
print "total traits, desired:\t", np.sum(np.sum(deploy_traits_desired))

# -----------------------------------------------------------------------------#
# initialize graph: all species move on same graph

graph = nx.grid_2d_graph(size_lattice, size_lattice) #, periodic = True)
# get the adjencency matrix
adjacency_m = nx.to_numpy_matrix(graph)
adjacency_m = np.squeeze(np.asarray(adjacency_m))


# -----------------------------------------------------------------------------#
# find optimal transition matrix

transition_m_init = optimal_transition_matrix(adjacency_m, deploy_robots_init, deploy_traits_desired, species_traits)
      
# -----------------------------------------------------------------------------#
# run microscopic stochastic simulation

t_max = 90
dt = 0.1
num_iter = 2

deploy_robots_micro = np.zeros((num_nodes, t_max, num_species, num_iter)) 
for it in range(num_iter): 
    print "Iteration: ", it
    robots_new, deploy_robots_micro[:,:,:,it] = microscopic_sim(t_max, dt, robots, deploy_robots_init, transition_m_init)
avg_deploy_robots_micro = np.mean(deploy_robots_micro,3)
   
# -----------------------------------------------------------------------------#
# run adaptive microscopic stochastic simulation


slices = 3
t_window = int(t_max / slices)

deploy_robots_micro_adapt = np.zeros((num_nodes, t_max, num_species, num_iter)) 


for it in range(num_iter): 
    transition_m = transition_m_init.copy()
    deploy_robots_init_slice = deploy_robots_init.copy()
    robots_slice = robots.copy()
    for sl in range(slices):
        print "Slice: ", sl
        robots_slice, deploy_robots_micro_slice = microscopic_sim(t_window, dt, robots_slice, deploy_robots_init_slice, transition_m)
        deploy_robots_init_slice = deploy_robots_micro_slice[:,-1,:]
        # put together slices
        deploy_robots_micro_adapt[:,sl*t_window:(sl+1)*t_window,:,it] = deploy_robots_micro_slice
        # calculate adapted transition matrix    
        transition_m = optimal_transition_matrix(adjacency_m, deploy_robots_init_slice, deploy_traits_desired, species_traits)

avg_deploy_robots_micro_adapt = np.mean(deploy_robots_micro_adapt,3)

# -----------------------------------------------------------------------------#
# euler integration
       
deploy_robots_euler = run_euler_integration_micro(t_max, dt, deploy_robots_init, transition_m_init)

# -----------------------------------------------------------------------------#
# plots


# plot evolution over time
species_index = 0
trait_index = 0

plot_robots_time(deploy_robots_euler, species_index)
plot_robots_time(avg_deploy_robots_micro, species_index)
plot_robots_time(avg_deploy_robots_micro_adapt, species_index)


#plot_traits_time(deploy_robots, species_traits, trait_index)
 
 







