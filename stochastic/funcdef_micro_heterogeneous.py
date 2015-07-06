# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:09:42 2015

@author: amanda
"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx

from optimize_transition_matrix_hetero import *



# -----------------------------------------------------------------------------#
# initialize robots

def initialize_robots(deploy_robots_init):
    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]
    
    total_num_robots = np.sum(np.sum(deploy_robots_init))
    robots = np.zeros((total_num_robots,2))
    robots_per_species = np.sum(deploy_robots_init,0)
    ind = 0
    # allocate species in column 0, nodes in column 1
    for s in range(num_species):
        robots[ind:ind+robots_per_species[s],0] = s
        ind_2 = ind
        for n in range(num_nodes):
            v = deploy_robots_init[n,s]
            robots[ind_2:ind_2+v,1] = n
            ind_2 += v
        ind += robots_per_species[s]
        
    return robots   



# -----------------------------------------------------------------------------#
# helper function that pick a random index given probabilities (summing to 1)

def pick_transition(p):
    rand = np.random.rand(1)
    v = 0
    for i in range(np.size(p)):
        v += p[i]
        if rand <= v:
            return i
    # Should not happen (unless probabilities do not sum to 1 exactly).
    return np.size(p) - 1


# -----------------------------------------------------------------------------#
# microscopic model

def microscopic_sim(t_max, dt, robots_init, deploy_robots_init, transition_m):

    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]
    
    deploy_robots = np.zeros((num_nodes, t_max, num_species))
    deploy_robots[:,0,:] = deploy_robots_init

    robots = robots_init.copy()

    # Pre-compute transition probabilities.
    ps = []
    for s in range(num_species):
        ks = transition_m[:,:,s] # transition rates
        ps.append(sp.linalg.expm(dt*ks)) # transition probabilities

    for t in range(1, t_max):
        # Propagate previous state.
        deploy_robots[:, t, :] = deploy_robots[:, t-1, :]

        for r in range(robots.shape[0]):
            r_s = robots[r,0] # robot species
            r_n = robots[r,1] # current node
            node = pick_transition(ps[int(r_s)][:,r_n])
            # update
            deploy_robots[r_n, t, r_s] -= 1
            deploy_robots[node, t, r_s]  += 1
            robots[r,1] = node

    return (robots, deploy_robots)

    
# -----------------------------------------------------------------------------#
# run euler integration and return time evolution

def run_euler_integration_micro(t_max, delta_t, deploy_robots_init, transition_m):
    
    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]
    

    deploy_robots = np.zeros((num_nodes,t_max, num_species))
    for s in range(num_species):
        for i in range(num_nodes):  
            deploy_robots[i,0,s] = deploy_robots_init[i,s]
            for t in range(1,t_max):
                deploy_robots[:,t,s] = deploy_robots[:,t-1,s] + delta_t*np.dot(transition_m[:,:,s], deploy_robots[:,t-1,s]) 
    
    return deploy_robots