# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 09:48:20 2015
@author: amanda
"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx
import sys
import pickle
import time

# my modules
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod

# -----------------------------------------------------------------------------#
# initialize world and robot community
save_data = False
save_plots = False
load_globals = True
save_globals = False
tstart = time.strftime("%Y%m%d-%H%M%S")
fix_species = True

# simulation parameters
t_max = 10.0 # influences desired state and optmization of transition matrix
t_max_sim = 8.0 # influences simulations and plotting
num_iter = 10 # iterations of micro sim
delta_t = 0.04 # time step
max_rate = 2.0 # Maximum rate possible for K.

# cost function
l_norm = 2 # 2: quadratic 1: absolute
match = 1 # 1: exact 0: at-least

if load_globals:
    graph = pickle.load(open("const_graph.p", "rb"))
    species_traits = pickle.load(open("const_species_traits.p", "rb"))
    deploy_robots_init = pickle.load(open("const_deploy_robots_init.p", "rb"))
    deploy_traits_init = pickle.load(open("const_deploy_traits_init.p", "rb"))
    deploy_traits_desired = pickle.load(open("const_deploy_traits_desired.p", "rb"))
    num_nodes = deploy_robots_init.shape[0]
    num_traits = species_traits.shape[1]
    num_species = species_traits.shape[0]
else:
    # create network of sites
    num_nodes = 10 
    # set of traits
    num_traits = 4
    max_trait_values = 2 # [0,1]: trait availability
    # robot species
    num_species = 3
    max_robots = 300 # maximum number of robots per node
    deploy_robots_init = np.random.randint(0, max_robots, size=(num_nodes, num_species))
    num_species = 3
    num_traits = 4

    if fix_species:
        num_species = 3
        num_traits = 4
        species_traits = np.array([[1,0,1,0],[1,0,0,1],[0,1,0,1]])
    else:
        # ensure each species has at least 1 trait, and that all traits are present
        species_traits = np.zeros((num_species, num_traits))
        while (min(np.sum(species_traits,0))==0 or min(np.sum(species_traits,1))==0):
            species_traits = np.random.randint(0, max_trait_values, (num_species, num_traits))
    # generate a random end state
    random_transition = random_transition_matrix(num_nodes, max_rate/2)  # Divide max_rate by 2 for the random matrix to give some slack.        
    # sample final desired trait distribution based on random transition matrix   
    deploy_robots_final = sample_final_robot_distribution(deploy_robots_init, random_transition, t_max*4., delta_t)
    deploy_traits_init = np.dot(deploy_robots_init, species_traits)
    deploy_traits_desired = np.dot(deploy_robots_final, species_traits)
    # if 'at-least' cost function, reduce number of traits desired
    if match==0:
        deploy_traits_desired *= (np.random.rand()*0.1 + 0.85)
        print "total traits, at least: \t", np.sum(np.sum(deploy_traits_desired))



print "total robots, init: \t", np.sum(np.sum(deploy_robots_init))
print "total traits, init: \t", np.sum(np.sum(deploy_traits_init))



# -----------------------------------------------------------------------------#
# initialize graph: all species move on same graph

if load_globals:
    graph = pickle.load(open("const_graph.p", "rb"))
else:
    graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.6)

# get the adjencency matrix
adjacency_m = nx.to_numpy_matrix(graph)
adjacency_m = np.squeeze(np.asarray(adjacency_m))


# -----------------------------------------------------------------------------#
# find optimal transition matrix

init_transition_values = np.array([])
transition_m = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init, deploy_traits_desired,
                                         species_traits, t_max, max_rate, l_norm, match, optimizing_t=True, force_steady_state=4.0)

transition_m_mac = transition_m.copy()


# -----------------------------------------------------------------------------#
# run microscopic stochastic simulation

num_timesteps = int(t_max_sim / delta_t)
# initialize robots
robots = initialize_robots(deploy_robots_init)

deploy_robots_micro = np.zeros((num_nodes, num_timesteps, num_species, num_iter))
print "Starting micro "
for it in range(num_iter):
    print "Iteration: ", it
    robots_new, deploy_robots_micro[:,:,:,it] = microscopic_sim(num_timesteps, delta_t, robots, deploy_robots_init, transition_m)
avg_deploy_robots_micro = np.mean(deploy_robots_micro,3)



# -----------------------------------------------------------------------------#
# run euler integration to drive robots to end state

deploy_robots_euler = run_euler_integration(deploy_robots_init, transition_m_mac, t_max_sim, delta_t)

deploy_robots_final = deploy_robots_euler[:,-1,:]
deploy_traits_final = np.dot(deploy_robots_final, species_traits) 

# -----------------------------------------------------------------------------#
# save variables

if save_data:
        
    tend = time.strftime("%Y%m%d-%H%M%S")
    prefix = "./data/micmac_V05_" # + tend + "_"
    print "Time start: ", tstart
    print "Time end: ", tend
    
    pickle.dump(species_traits, open(prefix+"st.p", "wb"))
    pickle.dump(graph, open(prefix+"graph.p", "wb"))
    pickle.dump(deploy_traits_init, open(prefix+"dti.p", "wb"))
    pickle.dump(deploy_traits_desired, open(prefix+"dtd.p", "wb"))
    pickle.dump(deploy_robots_micro, open(prefix+"drm.p", "wb"))
    pickle.dump(deploy_robots_euler, open(prefix+"dre.p", "wb"))

if save_globals:
    pickle.dump(species_traits, open("const_species_traits.p", "wb"))
    pickle.dump(graph, open("const_graph.p", "wb"))
    pickle.dump(deploy_robots_init, open("const_deploy_robots_init.p", "wb"))
    pickle.dump(deploy_traits_init, open("const_deploy_traits_init.p", "wb"))
    pickle.dump(deploy_traits_desired, open("const_deploy_traits_desired.p", "wb"))

# to read
# st = pickle.load(open(prefix+"st.p", "rb"))

# -----------------------------------------------------------------------------#
# plots

#nx.draw_circular(graph)
fig1 = nxmod.draw_circular(deploy_traits_init, graph, linewidths=3)
plt.show()
fig2 = nxmod.draw_circular(deploy_traits_final,graph, linewidths=3)
plt.show()
#nxmod.draw_circular(deploy_traits_desired,graph)
#plt.show()
   
# plot evolution over time
#species_index = 0
trait_index = 0
#plot_robots_time(avg_deploy_robots_micro, species_index) # plot micro average
#plot_robots_time(deploy_robots_euler, species_index) # plot macro
plot_traits_time(deploy_robots_euler, species_traits, trait_index)
    
# plot_robots_ratio_time_micmac(avg_deploy_robots_micro, deploy_robots_euler, deploy_robots_final, delta_t)
fig3 = plot_traits_ratio_time_micmac(deploy_robots_micro, deploy_robots_euler, deploy_traits_desired, 
                              species_traits, delta_t, match)

plt.show()
# -----------------------------------------------------------------------------#
# save plots
 
if save_plots:  
    fig1.savefig('./plots/gi.eps') 
    fig2.savefig('./plots/gd.eps')                           
    fig3.savefig('./plots/trt.eps')                             



