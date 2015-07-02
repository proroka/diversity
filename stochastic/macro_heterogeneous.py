# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:53:24 2015
@author: amandaprorok

"""




import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx

# my modules
from optimize_transition_matrix_hetero import *

# -----------------------------------------------------------------------------#

# 1. create graph
# 2. initialize robotdeployment on nodes
# 3. initialize transition matrix
# 4. run macro-discrete model 
# 5. plot evolution of robot population per node on graph


# -----------------------------------------------------------------------------#
# initialize world and robot community

# create network of sites
size_lattice = 2
num_nodes = size_lattice**2

# set of traits
num_traits = 3
max_trait_values = 2 # [0,1]: trait availability

# robot species
num_species = 3
# initial deployment is normalized number of robots per node
deploy_robots_init = np.zeros((num_nodes,num_species))
for s in range(num_species):
    deploy_robots_init[:,s] = np.random.rand(num_nodes)
    deploy_robots_init[:,s] /= np.sum(deploy_robots_init[:,s])

species_traits = np.random.randint(0, max_trait_values, (num_species, num_traits))
deploy_traits_init = np.dot(deploy_robots_init, species_traits)
deploy_traits_desired = np.random.rand(num_nodes, num_traits)

# -----------------------------------------------------------------------------#
# initialize graph: all species move on same graph

graph = nx.grid_2d_graph(size_lattice, size_lattice) #, periodic = True)
# get the adjencency matrix
adjacency_m = nx.to_numpy_matrix(graph)
adjacency_m = np.squeeze(np.asarray(adjacency_m))



# -----------------------------------------------------------------------------#
# find optimal transition matrix

find_optimal = True # optmize transition matrix for a given desired end state of traits per node
testing = True # use a simple testing scenario instread of random init

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
        
    # Specify the maximum time after which the initial state should reach the
    # desired state.
    max_time = 30
    # The basinhoping technique can optimize under bound constraints.
    # Fix the maximum transition rate.
    max_rate = 5    
                  # Optimize_Hetero(adjacency_matrix, initial_state, desired_steadystate, transform, max_time, max_rate, verbose):
    transition_m = Optimize_Hetero(adjacency_m, deploy_robots_init, deploy_traits_desired, species_traits, max_time, max_rate, verbose=True)
    print ''

else:
    # create random transition matrix
    transition_m = np.zeros((num_nodes,num_nodes,num_species))
    for i in range(num_species):    
        transition_m[:,:,i] = np.random.rand(num_nodes, num_nodes) * adjacency_m
        # k_ii is -sum(k_ij) s.t. sum(column)=0; ensures constant total number of robots
        np.fill_diagonal(transition_m[:,:,i], -np.sum(transition_m[:,:,i],0)) 
       
       
      
# -----------------------------------------------------------------------------#
# run euler integration to drive robots to end state

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

# draw graph with node size proportional to robot population
s_ind = 0;
scale = 1000 # scale size of node
print 'Initial configuration:'
#nx.draw_circular(graph, node_size=deploy_robots_init[:,s_ind]*scale)
nx.draw_circular(graph, node_size=np.sum(deploy_traits_init,1)*scale)

plt.show()       
# draw graph with node size proportional to robot population
print 'Final configuration:'
#nx.draw_circular(graph, node_size=deploy_robots[:,t_max-1,s_ind]*scale)
nx.draw_circular(graph, node_size=np.sum(deploy_traits_final,1)*scale)
plt.show()


# plot evolution of robot population over time
for n in range(num_nodes):
    x = np.arange(0, t_max)
    y = deploy_robots[n,:,s_ind]
    plt.plot(x,y)
    
plt.show()


"""
# -----------------------------------------------------------------------------#
# solve system analytically for steady-state

eig_values, eig_vectors = np.linalg.eig(transition_m)
i = np.argmax(eig_values) 
steady_state = eig_vectors[:,i].astype(float) 

steady_state /= sum(steady_state)  # Makes sure that the sum of all robots is 1.
diff = steady_state.T - deploy_robots[:,t_max-1]

print 'Final divergence (analytical vs euler): \n', diff

"""

# -----------------------------------------------------------------------------#
# solve system analytically for end time t_max


# analytic version
deploy_robots_final = np.zeros((num_nodes, num_species))
for s in range(num_species): # species index    
    A = transition_m[:,:,s].copy()
    x0 = deploy_robots_init[:,s].copy()
    deploy_robots_final[:,s] = np.dot(sp.linalg.expm(A*t_max), x0)
    #diff2 = x_tmax.T - deploy_robots[:,t_max-1,s_ind]
    #print 'Final divergence (exponential vs euler): \n', diff2


# final distribution of traits
deploy_traits_final = np.dot(deploy_robots_final,species_traits)







