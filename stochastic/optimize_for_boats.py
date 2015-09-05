# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:13:22 2015
@author: amandaprorok

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

t_max = 600.0       # 10 minutes (only influences the simulation plot).
max_rate = 0.01     # Maximum rate possible for K (1 / 100s == 0.01 Hz).
l_norm = 2          # 2: quadratic 1: absolute.
match = 1           # 1: exact 0: at-least

# Create graph (M = 4):
# 0 -- 2
# |    |
# 1 -- 3
graph = nx.Graph()
graph.add_nodes_from(i for i in range(4))
graph.add_edges_from([(0, 2), (2, 0),
                      (0, 1), (1, 0),
                      (1, 3), (3, 1),
                      (2, 3), (3, 2)])

# Get the adjencency matrix
adjacency_m = nx.to_numpy_matrix(graph)
adjacency_m = np.squeeze(np.asarray(adjacency_m))
print 'Adjacency matrix =\n', adjacency_m

# Create initial robot distribution (M x S).
# N=4
#deploy_robots_init = np.array([[2, 0],
#                               [0, 2],
#                               [0, 0],
#                               [0, 0]])
# N=6                               
#deploy_robots_init = np.array([[4, 0],
#                               [0, 2],
#                               [0, 0],
#                               [0, 0]])
# N=7                               
deploy_robots_init = np.array([[4, 0],
                               [0, 1],
                               [0, 0],
                               [0, 0]])
print 'Number of robots =', np.sum(deploy_robots_init)
print 'Number of robots per species =', np.sum(deploy_robots_init, axis=0)

# Create final robot distribution (M x S).
# N=4
#deploy_robots_final = np.array([[0, 0],
#                                [0, 0],
#                                [0, 2],
#                                [2, 0]])
# N=6
#deploy_robots_final = np.array([[0, 0],
#                                [0, 0],
#                                [0, 2],
#                                [4, 0]])
 # N=7
deploy_robots_final = np.array([[0, 0],
                                [0, 0],
                                [0, 1],
                                [4, 0]])                               
assert np.sum(np.abs(np.sum(deploy_robots_init, axis=0) - np.sum(deploy_robots_final, axis=0))) == 0, 'Number of robots is different between initial and final distribution'

# Create Q matrix (species_traits, S x U).
#species_traits = np.array([[1, 0],
 #                          [0, 1]]);

# redundant species                           
species_traits = np.array([[1, 0],
                           [1, 0]]);
                           
                           
print 'Species-Trait matrix =\n', species_traits

# Get trait distributions.
deploy_traits_init = np.dot(deploy_robots_init, species_traits)
deploy_traits_desired = np.dot(deploy_robots_final, species_traits)

# Optimize Ks.
init_transition_values = np.array([])
# To optimize, we need to tweak the input the params a bit.
# In particular, the params were tuned for max_rate = 2. and more robots.
Ks = optimal_transition_matrix(init_transition_values, adjacency_m, deploy_robots_init * 10., deploy_traits_desired * 10.,
                               species_traits, 10., 2., l_norm, match, optimizing_t=True, force_steady_state=1.)
Ks = Ks / 2. * max_rate  # Re-adjust the max_rate.

print '\nMatlab code to set up the K matrices:'
print '-------------------------------------\n'
print 'nspecies =', species_traits.shape[1]
print 'K = cell(nspecies);'
for s in range(species_traits.shape[1]):
    K = Ks[:, :, s]
    content = []
    rows = [K[i] for i in range(K.shape[0])]
    for row in rows:
        content.append(' '.join([str(v) for v in row]))
    print 'K{%d} = [%s];' % (s + 1, '; ...\n        '.join(content))
print ''

# Plot graph.
fig1 = nxmod.draw_circular(deploy_traits_init, graph, linewidths=3)
plt.axis('equal')
plt.show()
fig2 = nxmod.draw_circular(deploy_traits_desired, graph, linewidths=3)
plt.axis('equal')
plt.show()

def plot_traits_ratio_time_mac(deploy_robots_mac, deploy_traits_desired, transform, delta_t, match):
    fig = plt.figure()
    num_tsteps = deploy_robots_mac.shape[1]
    total_num_traits = np.sum(deploy_traits_desired)
    diffmac_rat = np.zeros(num_tsteps)
    for t in range(num_tsteps):
        if match==0:
            traits = np.dot(deploy_robots_mac[:,t,:], transform)
            diffmac = np.abs(np.minimum(traits - deploy_traits_desired, 0))
        else:
            traits = np.dot(deploy_robots_mac[:,t,:], transform)
            diffmac = np.abs(traits - deploy_traits_desired)
        diffmac_rat[t] = np.sum(diffmac) / total_num_traits
    x = np.arange(0, num_tsteps) * delta_t
    l2 = plt.plot(x, diffmac_rat, color='blue', linewidth=2, label='Macroscropic')
    plt.xlabel('Time [s]')
    plt.ylabel('Ratio of misplaced traits')
    return fig

# Simulate.
delta_t = 0.2
deploy_robots_euler = run_euler_integration(deploy_robots_init, Ks, t_max, delta_t)
fig3 = plot_traits_ratio_time_mac(deploy_robots_euler, deploy_traits_desired, species_traits, delta_t, match)
plt.show()
