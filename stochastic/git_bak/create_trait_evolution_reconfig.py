# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:43:36 2015
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
run = 'V24'

save_data = 1


save_globals = False
tstart = time.strftime("%Y%m%d-%H%M%S")
fix_species = True
fix_init = True
fix_final = True


# simulation parameters
t_max = 10.0 # influences desired state and optmization of transition matrix
t_max_sim = 8.0 # influences simulations and plotting
delta_t = 0.04 # time step
max_rate = 2.0 # Maximum rate possible for K.

# cost function
l_norm = 2 # 2: quadratic 1: absolute
match = 1 # 1: exact 0: at-least



# create network of sites
num_nodes = 8
# set of traits
num_traits = 4
max_trait_values = 2 # [0,1]: trait availability
# robot species
num_species = 3
max_robots = 300 # maximum number of robots per node
deploy_robots_init = np.random.randint(0, max_robots, size=(num_nodes, num_species))


num_species = 3
num_traits = 4
species_traits = np.array([[1,0,1,0],[1,0,0,1],[0,1,0,1]])

# -----------------------------------------------------------------------------#
# initialize graph: all species move on same graph


graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.3)

# get the adjencency matrix
adjacency_m = nx.to_numpy_matrix(graph)
adjacency_m = np.squeeze(np.asarray(adjacency_m))


# -----------------------------------------------------------------------------#
# generate configurations

# fix final
deploy_robots_final = np.random.randint(0, max_robots, size=(num_nodes, num_species))
deploy_robots_final_0 = deploy_robots_final.copy()
deploy_robots_final_1 = deploy_robots_final.copy()
deploy_robots_final_2 = deploy_robots_final.copy()

#deploy_robots_final_0[:2,:] = 2
#deploy_robots_final_1[2:4,:] = 2
#deploy_robots_final_2[4:,:] = 2

deploy_robots_final_0 = np.random.randint(0, max_robots, size=(num_nodes, num_species)).astype(float)
deploy_robots_final_1 = np.random.randint(0, max_robots, size=(num_nodes, num_species)).astype(float)
deploy_robots_final_2 = np.random.randint(0, max_robots, size=(num_nodes, num_species)).astype(float)


# fix initial
deploy_robots_init_0 = deploy_robots_init.copy()
deploy_robots_init_0[3:,:] = 0


sum_species = np.sum(deploy_robots_init_0,axis=0)


ts0 = np.sum(deploy_robots_final_0, axis=0)
ts1 = np.sum(deploy_robots_final_1, axis=0)
ts2 = np.sum(deploy_robots_final_2, axis=0)
for i in range(num_species):
    deploy_robots_final_0[:,i] = deploy_robots_final_0[:,i] / float(ts0[i]) * float(sum_species[i])
    deploy_robots_final_1[:,i] = deploy_robots_final_1[:,i] / float(ts1[i]) * float(sum_species[i])
    deploy_robots_final_2[:,i] = deploy_robots_final_2[:,i] / float(ts2[i]) * float(sum_species[i])
print np.sum(deploy_robots_final_0)
print np.sum(deploy_robots_final_1)
print np.sum(deploy_robots_final_2)

# allocate the following intial distribs
deploy_robots_init_1 = deploy_robots_final_0.copy()
deploy_robots_init_2 = deploy_robots_final_1.copy()

# calculate trait distribs
deploy_traits_init_0 = np.dot(deploy_robots_init_0, species_traits)
deploy_traits_desired_0 = np.dot(deploy_robots_final_0, species_traits)
deploy_traits_desired_1 = np.dot(deploy_robots_final_1, species_traits)
deploy_traits_desired_2 = np.dot(deploy_robots_final_2, species_traits)

# -----------------------------------------------------------------------------#
# find optimal transition matrix

init_transition_values = np.array([])

print "****"
print "Optimizing 1st transition matrix"
transition_m_0 = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init_0, deploy_traits_desired_0,
                                           species_traits, t_max, max_rate, l_norm, match, optimizing_t=True, force_steady_state=4.0)
print "****"
print "Running Euler for 1st transition matrix"
deploy_robots_euler_0 = run_euler_integration(deploy_robots_init_0, transition_m_0, t_max_sim, delta_t)
deploy_robots_init_1 = deploy_robots_euler_0[:,-1,:]
deploy_traits_init_1 = np.dot(deploy_robots_init_1, species_traits)

print "****"
print "Optimizing 2nd transition matrix"
transition_m_1 = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init_1, deploy_traits_desired_1,
                                           species_traits, t_max, max_rate, l_norm, match, optimizing_t=True, force_steady_state=4.0)
print "****"
print "Running Euler for 2nd transition matrix"
deploy_robots_euler_1 = run_euler_integration(deploy_robots_init_1, transition_m_1, t_max_sim, delta_t)
deploy_robots_init_2 = deploy_robots_euler_1[:,-1,:]
deploy_traits_init_2 = np.dot(deploy_robots_init_2, species_traits)

print "****"
print "Optimizing 3rd transition matrix"
transition_m_2 = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init_2, deploy_traits_desired_2,
                                           species_traits, t_max, max_rate, l_norm, match, optimizing_t=True, force_steady_state=4.0)
print "****"
print "Running Euler for 3rd transition matrix"
deploy_robots_euler_2 = run_euler_integration(deploy_robots_init_2, transition_m_2, t_max_sim, delta_t)


deploy_robots_evolution = np.concatenate((deploy_robots_euler_0, deploy_robots_euler_1, deploy_robots_euler_2),axis=1)


# -----------------------------------------------------------------------------#
# save variables

if save_data:

    tend = time.strftime("%Y%m%d-%H%M%S")
    prefix = "./data/" + run + "_micmac_"
    print "Time start: ", tstart
    print "Time end: ", tend

    pickle.dump(species_traits, open(prefix+"st.p", "wb"))
    pickle.dump(graph, open(prefix+"graph.p", "wb"))
    pickle.dump(deploy_robots_evolution, open(prefix+"drev.p", "wb"))

    pickle.dump(deploy_traits_init_0, open(prefix+"ti_0.p", "wb"))
    pickle.dump(deploy_traits_init_1, open(prefix+"ti_1.p", "wb"))
    pickle.dump(deploy_traits_init_2, open(prefix+"ti_2.p", "wb"))

    pickle.dump(deploy_traits_desired_0, open(prefix+"td_0.p", "wb"))
    pickle.dump(deploy_traits_desired_1, open(prefix+"td_1.p", "wb"))
    pickle.dump(deploy_traits_desired_2, open(prefix+"td_2.p", "wb"))






