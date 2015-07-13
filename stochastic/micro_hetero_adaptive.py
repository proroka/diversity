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
import sys

# my modules
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *


# -----------------------------------------------------------------------------#

# 1. create graph
# 2. initialize robot deployment on nodes
# 3. initialize transition matrix
# 4. run macro-discrete model  / micro simulator
# 5. plot evolution of robot population per node

# -----------------------------------------------------------------------------#
# initialize world and robot community

# simulation parameters
t_max = 10.0 # influences desired state and optmization of transition matrix
t_max_sim = 2.0 # influences simulations and plotting
num_iter = 2 # iterations of micro sim
delta_t = 0.02 # time step
max_rate = 5.0 # maximum transition rate possible for K.

# cost function
l_norm = 2 # 2: quadratic 1: absolute
match = 0 # 1: exact 0: at-least

# create network of sites
size_lattice = 2
num_nodes = size_lattice**2

# set of traits
num_traits = 3
max_trait_values = 2 # [0,1]: trait availability

# robot species
num_species = 2
max_robots = 10 # maximum number of robots per node
deploy_robots_init = np.random.randint(0, max_robots, size=(num_nodes, num_species))

species_traits = np.random.randint(0, max_trait_values, (num_species, num_traits))
deploy_traits_init = np.dot(deploy_robots_init, species_traits)

# random end state
random_transition = random_transition_matrix(num_nodes, max_rate / 2)  # Divide max_rate by 2 for the random matrix to give some slack.

# sample final desired trait distribution based on random transition matrix
deploy_robots_final = sample_final_robot_distribution(deploy_robots_init, random_transition, t_max, delta_t)
deploy_traits_desired = np.dot(deploy_robots_final, species_traits)

# initialize robots
robots = initialize_robots(deploy_robots_init)

# Set to True, to just run the optimization.
just_optimize = False


print "total robots, init: \t", np.sum(np.sum(deploy_robots_init))
print "total traits, init: \t", np.sum(np.sum(deploy_traits_init))

# if 'at-least' cost function, reduce number of traits desired
if match==0:
    deploy_traits_desired *= (np.random.rand()*0.1 + 0.85)
    print "total traits, at least: \t", np.sum(np.sum(deploy_traits_desired))

# -----------------------------------------------------------------------------#
# initialize graph: all species move on same graph

# graph = nx.grid_2d_graph(size_lattice, size_lattice) #, periodic = True)
# Another option for the graph: nx.connected_watts_strogatz_graph(num_nodes, num_nodes - 1, 0.5)
graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.5)

# get the adjencency matrix
adjacency_m = nx.to_numpy_matrix(graph)
adjacency_m = np.squeeze(np.asarray(adjacency_m))


# -----------------------------------------------------------------------------#
# find optimal transition matrix for plain micro

transition_m_init = optimal_transition_matrix(adjacency_m, deploy_robots_init, deploy_traits_desired,
                                              species_traits, t_max, max_rate,l_norm, match, optimizing_t=True, force_steady_state=10.0)

# -----------------------------------------------------------------------------#
# run microscopic stochastic simulation

num_timesteps = int(t_max_sim / delta_t)

deploy_robots_micro = np.zeros((num_nodes, num_timesteps, num_species, num_iter))
for it in range(num_iter):
    print "Iteration: ", it
    robots_new, deploy_robots_micro[:,:,:,it] = microscopic_sim(num_timesteps, delta_t, robots, deploy_robots_init, transition_m_init)
avg_deploy_robots_micro = np.mean(deploy_robots_micro,3)

  
# -----------------------------------------------------------------------------#
# run adaptive microscopic stochastic simulation, RHC

slices = 5
t_window = t_max_sim / slices
numts_window = int(t_window / delta_t)

deploy_robots_micro_adapt = np.zeros((num_nodes, num_timesteps, num_species, num_iter)) 


for it in range(num_iter): 
    transition_m = transition_m_init.copy()
    deploy_robots_init_slice = deploy_robots_init.copy()
    robots_slice = robots.copy()
    for sl in range(slices):
        print "Slice: ", sl
        robots_slice, deploy_robots_micro_slice = microscopic_sim(numts_window, delta_t, robots_slice, deploy_robots_init_slice, transition_m)
        deploy_robots_init_slice = deploy_robots_micro_slice[:,-1,:]
        # put together slices
        deploy_robots_micro_adapt[:,sl*numts_window:(sl+1)*numts_window,:,it] = deploy_robots_micro_slice
        # calculate adapted transition matrix    
        transition_m = optimal_transition_matrix(adjacency_m, deploy_robots_init_slice, deploy_traits_desired, 
                                                 species_traits, t_max, max_rate,l_norm, match, optimizing_t=True, force_steady_state=t_window)

avg_deploy_robots_micro_adapt = np.mean(deploy_robots_micro_adapt,3)


# -----------------------------------------------------------------------------#
# euler integration

deploy_robots_euler = run_euler_integration(deploy_robots_init, transition_m_init, t_max_sim, delta_t)


# -----------------------------------------------------------------------------#
# plots

# plot graph
nxmod.draw_circular(deploy_traits_init,graph)
plt.show()
nxmod.draw_circular(deploy_traits_final,graph)
plt.show()

# plot traits ratio
plot_traits_ratio_time_micmac(avg_deploy_robots_micro,deploy_robots_euler, deploy_traits_desired, 
                              species_traits, delta_t, match)
                              
plot_traits_ratio_time_micmac(avg_deploy_robots_micro, avg_deploy_robots_micro_adapt, deploy_traits_desired, 
                              species_traits, delta_t, match)





