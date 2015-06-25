# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:43:50 2015
@author: amandaprorok

"""


import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx

# my modules
import funcdef_micro as fd


# -----------------------------------------------------------------------------#
# 1. create graph
# 2. initialize robotdeployment on nodes
# 3. initialize transition matrix
# 4. run macro-discrete model 
# 5. plot evolution of robot population per node on graph


# create 2d lattice graph
size_lattice = 3
num_nodes = size_lattice**2
graph = nx.grid_2d_graph(size_lattice, size_lattice) #, periodic = True)
# get the adjencency matrix
adjacency_m = nx.to_numpy_matrix(graph)
adjacency_m = np.squeeze(np.asarray(adjacency_m))

# initialize robot population, random node allocation
num_robots = 10
state_robots_init = np.random.randint(0,num_nodes,(num_robots,1))
state_robots_init = state_robots_init.astype(float)
# initial deployment is normalized number of robots per node
deploy_robots_init = np.zeros((num_nodes,1))
for i in range(num_nodes):
    deploy_robots_init[i] = np.size(np.where(state_robots_init == i))
deploy_robots_init /= np.sum(deploy_robots_init)

# draw graph with node size proportional to robot population
scale = 1000 # scale size of node
print 'Initial configuration:'
nx.draw_circular(graph, node_size=deploy_robots_init*scale)
plt.show()

# create random transition matrix
transition_m_init = np.random.rand(num_nodes, num_nodes) * adjacency_m
#transition_m_init = np.array([[0.,0.8,0.,0],[0.,0.,0.8,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
transition_m = transition_m_init.copy()
# k_ii is -sum(k_ij) s.t. sum(column)=0; ensures constant total number of robots
np.fill_diagonal(transition_m, -np.sum(transition_m_init,0)) 

# state machine, agents
t_max = 100
delta_t = 0.1

deploy_robots = np.zeros((num_nodes,t_max))
for i in range(num_nodes):
    deploy_robots[i,0] = deploy_robots_init[i]
for t in range(1,t_max):
    deploy_robots[:,t] = deploy_robots[:,t-1] + delta_t*np.dot(transition_m, deploy_robots[:,t-1])
        
# draw graph with node size proportional to robot population
print 'Final configuration:'
nx.draw_circular(graph, node_size=deploy_robots[:,t_max-1]*scale)
plt.show()

# plot evolution of robot population over time
for n in range(num_nodes):
    x = np.arange(0,t_max)
    y = deploy_robots[n,:]
    plt.plot(x,y)
    
plt.show()














