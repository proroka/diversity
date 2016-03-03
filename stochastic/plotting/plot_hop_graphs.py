import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import pickle

# my modules
sys.path.append('../utilities')
sys.path.append('..')
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod
from generate_Q import *
from simple_orrank import *


# returns a list of neighbors for each node
def GetNeighbors(A, n_hops):
    t = np.identity(A.shape[0])
    for _ in range(n_hops):
        t += t.dot(A)
    neighbors = []
    for i in range(A.shape[0]):
        neighbors.append(np.where(t[i,:] > 0)[0].tolist())
    return neighbors


def BuildLocalDistribution(B, node_index, neighbors):
    num_nodes = B.shape[0]
    deploy = B.copy()
    # Set to zero all neighbors.
    deploy[neighbors[node_index],:] = 0
    # Average all robots over the non-neighboring nodes.
    uniform = np.ones((num_nodes, num_species)) * (np.sum(deploy, 0) / float(num_nodes - len(neighbors[node_index])))
    # Put the original robot counts in the neighboring nodes.
    uniform[neighbors[node_index],:] = B[neighbors[node_index],:]
    return uniform

num_species = 4
num_traits = 5
num_nodes = 8
total_num_robots = 100
node_of_interest = 0

# Build graph.
graph = nx.connected_watts_strogatz_graph(num_nodes, 3, 0.6)
adjacency_m = nx.to_numpy_matrix(graph)
adjacency_m = np.squeeze(np.asarray(adjacency_m))

# Build trait-species matrix.
rk = 0
while rk != num_species:
    species_traits, rk, s = generate_Q(num_species, num_traits)

# Build trait distribution.
deploy_robots_init = np.random.randint(0, 100, size=(num_nodes, num_species))
deploy_robots_init = deploy_robots_init * total_num_robots / np.sum(np.sum(deploy_robots_init, axis=0))
sum_species = np.sum(deploy_robots_init,axis=0)
deploy_traits_init = np.dot(deploy_robots_init, species_traits)

fig = plt.figure()
nxmod.draw_circular(deploy_robots_init, graph, linewidths=3)
plt.axis('equal')
plt.title('nhops = infinity')
plt.show()
fig.savefig('./plots/nhops_all.eps')

for nhops in range(5):
    neighbor_nodes = GetNeighbors(adjacency_m, nhops)
    deploy_robots_init_hops = BuildLocalDistribution(deploy_robots_init, node_of_interest, neighbor_nodes)
    fig = plt.figure()
    nxmod.draw_circular(deploy_robots_init_hops, graph, linewidths=3)
    plt.axis('equal')
    plt.title('nhops = %d' % nhops)
    plt.show()
    fig.savefig('./plots/nhops_%d.eps' % nhops)
