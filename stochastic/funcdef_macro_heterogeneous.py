# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:33:04 2015
@author: amanda

"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx

from optimize_transition_matrix_hetero_fast_gradient import *


# -----------------------------------------------------------------------------#
# sample a random transition matrix

def random_transition_matrix(num_nodes):

    random_transition = np.random.rand(num_nodes, num_nodes) # * adjacency_m
    # k_ii is -sum(k_ij) s.t. sum(column)=0; ensures constant total number of robots
    np.fill_diagonal(random_transition, np.zeros(num_nodes))
    np.fill_diagonal(random_transition, -np.sum(random_transition,0))

    return random_transition


# -----------------------------------------------------------------------------#
# run euler integration to sample random end state

def sample_final_robot_distribution(deploy_robots_init, random_transition, t_max, delta_t):
    
    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]

    t_max = 50
    delta_t = 0.01
    deploy_robots_sample = np.zeros((num_nodes,t_max, num_species))
    for s in range(num_species):
        for i in range(num_nodes):  
            deploy_robots_sample[i,0,s] = deploy_robots_init[i,s]
            for t in range(1,t_max):
                deploy_robots_sample[:,t,s] = deploy_robots_sample[:,t-1,s] + delta_t*np.dot(random_transition, deploy_robots_sample[:,t-1,s])
    
    return deploy_robots_sample[:,t_max-1,:]
    
    
    
# -----------------------------------------------------------------------------#
# run euler integration and return time evolution
    
def run_euler_integration(deploy_robots_init, transition_m, t_max, delta_t):

    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]

    t_max = 50
    delta_t = 0.01
    deploy_robots = np.zeros((num_nodes,t_max, num_species))
    for s in range(num_species):
        deploy_robots[:,0,s] = deploy_robots_init[:,s]
        for t in range(1,t_max):
            deploy_robots[:,t,s] = deploy_robots[:,t-1,s] + delta_t*np.dot(transition_m[:,:,s], deploy_robots[:,t-1,s])

    return deploy_robots
    
# -----------------------------------------------------------------------------#
# find optimal transition matrix

def optimal_transition_matrix(adjacency_m, deploy_robots_init, deploy_traits_desired, species_traits, max_time, max_rate):
    
    find_optimal = True
   
    # Specify the maximum time after which the initial state should reach the
    # desired state.
    #max_time = 100
    # The basinhoping technique can optimize under bound constraints.
    # Fix the maximum transition rate.
    #max_rate = 5    

    verbose = True
    if find_optimal:                  
        transition_m = Optimize_Hetero_Fast(adjacency_m, deploy_robots_init, deploy_traits_desired, species_traits, max_time, max_rate, verbose)

    else:
        # create random transition matrix
        num_nodes = adjacency_m.shape[0]
        num_species = species_traits.shape[0]
        transition_m = np.zeros((num_nodes,num_nodes,num_species))
        for i in range(num_species):    
            transition_m[:,:,i] = np.random.rand(num_nodes, num_nodes) * adjacency_m
            # k_ii is -sum(k_ij) s.t. sum(column)=0; ensures constant total number of robots
            np.fill_diagonal(transition_m[:,:,i], -np.sum(transition_m[:,:,i],0)) 
           
    return transition_m       
    
    
# -----------------------------------------------------------------------------#
# plot network  
# adds all traits to scale size of nodes

def plot_network(graph, deploy_traits_init, deploy_traits_final):
    # draw graph with node size proportional to robot population
    scale = 10 # scale size of node
    
    print 'Initial configuration:'
    #nx.draw_circular(graph, node_size=deploy_robots_init[:,s_ind]*scale)
    nx.draw(graph)    
    plt.show()
    
    dg = nx.DiGraph(graph)
    nx.draw(dg)    
    
    #nx.draw_circular(graph, node_size=np.sum(deploy_traits_init,1)*scale)
    #nx. draw_networkx_labels(G, pos[, labels, ...])
    
    plt.show()       
    
    # draw graph with node size proportional to robot population
#    print 'Final configuration:'
    #nx.draw_circular(graph, node_size=deploy_robots[:,t_max-1,s_ind]*scale)
 #   nx.draw_circular(graph, node_size=np.sum(deploy_traits_final,1)*scale)
  #  plt.show()



# -----------------------------------------------------------------------------#
# plot 

def plot_robots_time(deploy_robots, species_ind):

    num_nodes = deploy_robots.shape[0]
    t_max = deploy_robots.shape[1]
    
    # plot evolution of robot population over time
    for n in range(num_nodes):
        x = np.arange(0, t_max)
        y = deploy_robots[n,:,species_ind]
        plt.plot(x,y)
    plt.show()    

# -----------------------------------------------------------------------------#
# plot 

def plot_traits_time(deploy_robots, species_traits, trait_ind):

    num_nodes = deploy_robots.shape[0]
    t_max = deploy_robots.shape[1]
    num_traits = species_traits.shape[1]
    
    deploy_traits = np.zeros((num_nodes, t_max, num_traits))
    for t in range(t_max):    
        deploy_traits[:,t,:] = np.dot(deploy_robots[:,t,:], species_traits)
    
    # plot evolution of trait distribution over time
    for n in range(num_nodes):
        x = np.arange(0, t_max)
        y = deploy_traits[n,:, trait_ind]
        plt.plot(x,y)
    plt.show()    


# -----------------------------------------------------------------------------#
# this doesnt work 

 
def clear():
    os.system('cls')
    return None
    

def clear_all():
    #clear()   
    gl = globals().copy
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
    
        del globals()[var]
   
    
    