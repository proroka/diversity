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
run = 'B02'

save_data = True
load_graph = True
save_graph = False

tstart = time.strftime("%Y%m%d-%H%M%S")
fix_species = True
fix_init = True
fix_final = True
do_optimize = True

# simulation parameters
t_max = 8.0 # influences desired state and optmization of transition matrix
t_max_sim = 7.0 # influences simulations and plotting
delta_t = 0.04 # time step
max_rate = 1.0 # Maximum rate possible for K.

# cost function
l_norm = 2 # 2: quadratic 1: absolute
match = 1 # 1: exact 0: at-least
FSS = 4.0

# create network of sites
num_nodes = 10

# robot species
total_num_robots = 100.0 
num_species = 3
num_traits = 4

species_traits = np.array([[1,0,1,0],[1,0,0,1],[0,1,0,1]])
sum_species = np.array([30.0 ,10.0 ,60.0])

deploy_robots_init = np.random.randint(0, 100, size=(num_nodes, num_species)).astype(float)

# normalize initial distribution
ts = np.sum(deploy_robots_init, axis=0)
for i in range(num_species):
    deploy_robots_init[:,i] = deploy_robots_init[:,i] / float(ts[i]) * float(sum_species[i])

sum_species_t = np.sum(deploy_robots_init,axis=0)
print 'init_0', sum_species_t

# -----------------------------------------------------------------------------#
# initialize graph: all species move on same graph

if load_graph:
    graph = pickle.load(open("./data/const_evolution_graph.p", "rb"))
    adjacency_m = pickle.load(open("./data/const_evolution_adjacency.p", "rb"))
    
else:
    graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.3)
    # get the adjencency matrix
    adjacency_m = nx.to_numpy_matrix(graph)
    adjacency_m = np.squeeze(np.asarray(adjacency_m))



# -----------------------------------------------------------------------------#
# generate configurations

# fix final
deploy_robots_final = np.random.randint(0, 100, size=(num_nodes, num_species))
deploy_robots_final_0 = deploy_robots_final.copy().astype(float)
deploy_robots_final_1 = deploy_robots_final.copy().astype(float)
deploy_robots_final_2 = deploy_robots_final.copy().astype(float)


# Adjust distributions
ff = 0.15
deploy_robots_init_0 = deploy_robots_init.copy()
ind = range(3,10) # robots in 0,1,2
deploy_robots_init_0[ind,:] = deploy_robots_init_0[ind,:] * ff
ind = [0,1,2,5,6,7,8,9] # robots in 3,4
deploy_robots_final_0[ind,:] = deploy_robots_final_0[ind,:] * ff
ind = [0,1,2,3,4,7,8,9] # robots in 5,6
deploy_robots_final_1[ind,:] = deploy_robots_final_1[ind,:] * ff
ind = range(0,7) # robots in 7,8,9
deploy_robots_final_2[ind,:] = deploy_robots_final_2[ind,:] * ff


# normalize final distribution
tsi = np.sum(deploy_robots_init_0, axis=0)
ts0 = np.sum(deploy_robots_final_0, axis=0)
ts1 = np.sum(deploy_robots_final_1, axis=0)
ts2 = np.sum(deploy_robots_final_2, axis=0)
for i in range(num_species):
    deploy_robots_init_0[:,i] = deploy_robots_init_0[:,i] / float(tsi[i]) * float(sum_species[i])
    deploy_robots_final_0[:,i] = deploy_robots_final_0[:,i] / float(ts0[i]) * float(sum_species[i])
    deploy_robots_final_1[:,i] = deploy_robots_final_1[:,i] / float(ts1[i]) * float(sum_species[i])
    deploy_robots_final_2[:,i] = deploy_robots_final_2[:,i] / float(ts2[i]) * float(sum_species[i])
print 'init_0', np.sum(deploy_robots_init_0, axis=0)
print 'final_0', np.sum(deploy_robots_final_0, axis=0)
print 'final_1', np.sum(deploy_robots_final_1, axis=0)
print 'final_2', np.sum(deploy_robots_final_2, axis=0)

# allocate the following intial distribs as previous final
deploy_robots_init_1 = deploy_robots_final_0.copy()
deploy_robots_init_2 = deploy_robots_final_1.copy()

# calculate trait distribs
deploy_traits_init_0 = np.dot(deploy_robots_init_0, species_traits)
deploy_traits_desired_0 = np.dot(deploy_robots_final_0, species_traits)
deploy_traits_desired_1 = np.dot(deploy_robots_final_1, species_traits)
deploy_traits_desired_2 = np.dot(deploy_robots_final_2, species_traits)

# -----------------------------------------------------------------------------#
# plot graph


nxmod.draw_circular(deploy_traits_desired_0,graph, linewidths=3)
plt.axis('equal')
plt.show()


# -----------------------------------------------------------------------------#
# find optimal transition matrix

init_transition_values = np.array([])

if do_optimize:

    print "****"
    print "Optimizing 1st transition matrix"
    transition_m_0 = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init_0, deploy_traits_desired_0,
                                               species_traits, t_max, max_rate, l_norm, match, optimizing_t=True, force_steady_state=FSS)
    print "****"
    print "Running Euler for 1st transition matrix"
    deploy_robots_euler_0 = run_euler_integration(deploy_robots_init_0, transition_m_0, t_max_sim, delta_t)
    deploy_robots_init_1 = deploy_robots_euler_0[:,-1,:]
    deploy_traits_init_1 = np.dot(deploy_robots_init_1, species_traits)
    
    print "****"
    print "Optimizing 2nd transition matrix"
    transition_m_1 = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init_1, deploy_traits_desired_1,
                                               species_traits, t_max, max_rate, l_norm, match, optimizing_t=True, force_steady_state=FSS)
    print "****"
    print "Running Euler for 2nd transition matrix"
    deploy_robots_euler_1 = run_euler_integration(deploy_robots_init_1, transition_m_1, t_max_sim, delta_t)
    deploy_robots_init_2 = deploy_robots_euler_1[:,-1,:]
    deploy_traits_init_2 = np.dot(deploy_robots_init_2, species_traits)
    
    print "****"
    print "Optimizing 3rd transition matrix"
    transition_m_2 = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init_2, deploy_traits_desired_2,
                                               species_traits, t_max, max_rate, l_norm, match, optimizing_t=True, force_steady_state=FSS)
    print "****"
    print "Running Euler for 3rd transition matrix"
    deploy_robots_euler_2 = run_euler_integration(deploy_robots_init_2, transition_m_2, t_max_sim, delta_t)
    
    
    deploy_robots_evolution = np.concatenate((deploy_robots_euler_0, deploy_robots_euler_1, deploy_robots_euler_2),axis=1)


# -----------------------------------------------------------------------------#
# Print transition matrix for matlab

def print_K(Ks, species_traits):
    print '\nMatlab code to set up the K matrices:'
    print '-------------------------------------\n'
    print 'nspecies =', species_traits.shape[0]
    print 'K = cell(nspecies);'
    for s in range(species_traits.shape[0]):
        K = Ks[:, :, s]
        content = []
        rows = [K[i] for i in range(K.shape[0])]
        for row in rows:
            content.append(' '.join([str(v) for v in row]))
        print 'K{%d} = [%s];' % (s + 1, '; ...\n        '.join(content))
    print ''


# -----------------------------------------------------------------------------#
# Print transition matrices for matlab simulation

if do_optimize:
    print_K(transition_m_0, species_traits)
    print_K(transition_m_1, species_traits)
    print_K(transition_m_2, species_traits)

# -----------------------------------------------------------------------------#
# save variables

if save_data:

    tend = time.strftime("%Y%m%d-%H%M%S")
    prefix = "./data/" + run + "_evolution_"
    print "Time start: ", tstart
    print "Time end: ", tend

    pickle.dump(sum_species, open(prefix+"ss.p", "wb"))
    pickle.dump(adjacency_m, open(prefix+"adj.p", "wb"))

    pickle.dump(species_traits, open(prefix+"st.p", "wb"))
    pickle.dump(graph, open(prefix+"graph.p", "wb"))
    

    pickle.dump(deploy_traits_init_0, open(prefix+"ti_0.p", "wb"))
    pickle.dump(deploy_traits_init_1, open(prefix+"ti_1.p", "wb"))
    pickle.dump(deploy_traits_init_2, open(prefix+"ti_2.p", "wb"))

    pickle.dump(deploy_traits_desired_0, open(prefix+"td_0.p", "wb"))
    pickle.dump(deploy_traits_desired_1, open(prefix+"td_1.p", "wb"))
    pickle.dump(deploy_traits_desired_2, open(prefix+"td_2.p", "wb"))
    
    pickle.dump(deploy_robots_evolution, open(prefix+"drev.p", "wb"))
    
    
if save_graph:
    
    pickle.dump(adjacency_m, open("./data/const_evolution_adjacency.p", "wb"))
    pickle.dump(graph, open("./data/const_evolution_graph.p", "wb"))



