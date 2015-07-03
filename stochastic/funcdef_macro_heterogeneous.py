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

from optimize_transition_matrix_hetero import *


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

def sample_final_robot_distribution(deploy_robots_init, random_transition):
    
    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]

    t_max = 50
    delta_t = 0.1
    deploy_robots_sample = np.zeros((num_nodes,t_max, num_species))
    for s in range(num_species):
        for i in range(num_nodes):  
            deploy_robots_sample[i,0,s] = deploy_robots_init[i,s]
            for t in range(1,t_max):
                deploy_robots_sample[:,t,s] = deploy_robots_sample[:,t-1,s] + delta_t*np.dot(random_transition, deploy_robots_sample[:,t-1,s])
    
    return deploy_robots_sample[:,t_max-1,:]
    
    
    
# -----------------------------------------------------------------------------#
# run euler integration to sample random end state

def run_euler_integration(deploy_robots_init, transition_m, species_traits):
    
    plotting = False
    
    num_nodes = deploy_robots_init.shape[0]
    num_species = deploy_robots_init.shape[1]
    
    t_max = 50
    delta_t = 0.1
    deploy_robots = np.zeros((num_nodes,t_max, num_species))
    for s in range(num_species):
        for i in range(num_nodes):  
            deploy_robots[i,0,s] = deploy_robots_init[i,s]
            for t in range(1,t_max):
                deploy_robots[:,t,s] = deploy_robots[:,t-1,s] + delta_t*np.dot(transition_m[:,:,s], deploy_robots[:,t-1,s])
    
    deploy_robots_final = deploy_robots[:,t_max-1,:]
    deploy_traits_final = np.dot(deploy_robots_final, species_traits)   
    
    if plotting:
        # plot evolution of robot population over time
        for n in range(num_nodes):
            x = np.arange(0, t_max)
            y = deploy_robots[n,:,s_ind]
            plt.plot(x,y)
        plt.show()    
    
    
    return (deploy_robots_final, deploy_traits_final)       
    
    
# -----------------------------------------------------------------------------#
# find optimal transition matrix

def optimal_transition_matrix(adjacency_m, deploy_robots_init, deploy_traits_desired, species_traits):
    find_optimal = True
    """
    if find_optimal:
    # for testing
    if (testing and num_nodes==4 and num_species==3 and num_traits==3):
        print "\n *** Testing *** \n"
        deploy_robots_init = np.array([[1., 1., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]) # all species in node 1
        species_traits = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]) # each species has 1 complementary trait
        deploy_traits_init = np.array([[1., 1., 1,], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]) # all traits in node 1
        deploy_traits_desired = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [1., 1., 1.]]) # all traits in node 4
    else:
        # desired deployment
        deploy_traits_desired = np.random.rand(num_nodes,num_traits)
    """
    # Specify the maximum time after which the initial state should reach the
    # desired state.
    max_time = 30
    # The basinhoping technique can optimize under bound constraints.
    # Fix the maximum transition rate.
    max_rate = 5    

    verbose = True
    if find_optimal:                  
        transition_m = Optimize_Hetero(adjacency_m, deploy_robots_init, deploy_traits_desired, species_traits, max_time, max_rate, verbose)

    else:
        # create random transition matrix
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
    nx.draw_circular(graph, node_size=np.sum(deploy_traits_init,1)*scale)
    
    plt.show()       
    # draw graph with node size proportional to robot population
    print 'Final configuration:'
    #nx.draw_circular(graph, node_size=deploy_robots[:,t_max-1,s_ind]*scale)
    nx.draw_circular(graph, node_size=np.sum(deploy_traits_final,1)*scale)
    plt.show()



# -----------------------------------------------------------------------------#
# this doesnt work: 

 
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
   
    
    