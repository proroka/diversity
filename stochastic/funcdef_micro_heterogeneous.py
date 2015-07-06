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
# microscopic model

def microscopic_sim(t_max, dt, robots, deploy_robots_init, transition_m):

    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]
    
    deploy_robots = np.zeros((num_nodes, t_max, num_species))
    deploy_robots[:,0,:] = deploy_robots_init

    for t in range(1, t_max):
        for r in range(robots.shape[0]):
            r_s = robots[r,0] # robot species
            r_n = robots[r,1] # current node
            # outgoing rates
            ks = transition_m[:,:,r_s] # transition rates
            ps = sp.linalg.expm(dt*ks) # transition probabilities
            rands = np.random.rand(num_nodes)
            activated_edges = rands < ps[:,r_n] # column indicates current node
            a = np.nonzero(activated_edges) # returns a tuple of indices
            if np.sum(a[0]) == 1: # choose this transition
                node = a[0][0]
                robots[r,1] = node
            if np.sum(a[0]) > 1:  # choose random transition
                node = a[0][np.random.randint(np.size(a[0]))]
                robots[r,1] = node
            if np.sum(a[0]) > 0: # update summary representation if robot transitions
                deploy_robots[r_n, t, r_s] = deploy_robots[r_n, t-1, r_s] - 1         
                deploy_robots[node, t, r_s]  = deploy_robots[node, t-1, r_s] + 1
            else: # propagate previous state
                deploy_robots[r_n, t, r_s] = deploy_robots[r_n, t-1, r_s]

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