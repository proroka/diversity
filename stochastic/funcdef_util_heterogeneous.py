# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:26:07 2015
@author: amanda

"""

import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx


# -----------------------------------------------------------------------------#
# plot ratio of desired vs actual robot distribution

def plot_robots_ratio_time_micmac(deploy_robots_mic, deploy_robots_mac, deploy_robots_desired):
    
    num_iter = deploy_robots_mic.shape[1]

    diffmic_sqs = np.zeros(num_iter)
    diffmac_sqs = np.zeros(num_iter)
    for t in range(num_iter):
        diffmic = deploy_robots_mic[:,t,:] - deploy_robots_desired
        diffmic_sqs[t] = np.sum(np.square(diffmic))
        diffmac = deploy_robots_mac[:,t,:] - deploy_robots_desired
        diffmac_sqs[t] = np.sum(np.square(diffmac))
        
    x = np.arange(0, num_iter)
    plt.plot(x,diffmic_sqs)
    plt.plot(x,diffmac_sqs)
        
    plt.show()
    
# -----------------------------------------------------------------------------#
# plot ratio of desired vs actual robot distribution

def plot_traits_ratio_time_micmac(deploy_robots_mic, deploy_robots_mac, deploy_traits_desired, transform):
    
    num_iter = deploy_robots_mic.shape[1]

    diffmic_sqs = np.zeros(num_iter)
    diffmac_sqs = np.zeros(num_iter)
    for t in range(num_iter):
        traits = np.dot(deploy_robots_mic[:,t,:], transform)
        diffmic = traits - deploy_traits_desired
        diffmic_sqs[t] = np.sum(np.square(diffmic))
        
        traits = np.dot(deploy_robots_mac[:,t,:], transform)
        diffmac =  traits - deploy_traits_desired
        diffmac_sqs[t] = np.sum(np.square(diffmac))
        
    x = np.arange(0, num_iter)
    plt.plot(x,diffmic_sqs)
    plt.plot(x,diffmac_sqs)
        
    plt.show()  
    
    
# -----------------------------------------------------------------------------#
# plot ratio of desired vs actual robot distribution

def plot_robots_ratio_time(deploy_robots, deploy_robots_desired):
    
    num_iter = deploy_robots.shape[1]

    diffsqs = np.zeros(num_iter)
    for t in range(num_iter):
        diff = deploy_robots[:,t,:] - deploy_robots_desired
        diffsqs[t] = np.sum(np.square(diff))

    x = np.arange(0, num_iter)
    plt.plot(x,diffsqs)
    plt.show()
    
# -----------------------------------------------------------------------------#
# plot
    
def plot_traits_ratio_time(deploy_robots, deploy_traits_desired, transform):
    
    num_iter = deploy_robots.shape[1]

    diffsqs = np.zeros(num_iter)
    for t in range(num_iter):
        traits = np.dot(deploy_robots[:,t,:], transform)
        diff = traits - deploy_traits_desired
        diffsqs[t] = np.sum(np.square(diff))

    x = np.arange(0, num_iter)
    plt.plot(x,diffsqs)
    plt.show()




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
    num_iter = deploy_robots.shape[1]

    # plot evolution of robot population over time
    for n in range(num_nodes):
        x = np.arange(0, num_iter)
        y = deploy_robots[n,:,species_ind]
        plt.plot(x,y)
    plt.show()

# -----------------------------------------------------------------------------#
# plot

def plot_traits_time(deploy_robots, species_traits, trait_ind):

    num_nodes = deploy_robots.shape[0]
    num_iter = deploy_robots.shape[1]
    num_traits = species_traits.shape[1]

    deploy_traits = np.zeros((num_nodes, num_iter, num_traits))
    for t in range(num_iter):
        deploy_traits[:,t,:] = np.dot(deploy_robots[:,t,:], species_traits)

    # plot evolution of trait distribution over time
    for n in range(num_nodes):
        x = np.arange(0, num_iter)
        y = deploy_traits[n,:, trait_ind]
        plt.plot(x,y)
    plt.show()



"""
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


""" 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    