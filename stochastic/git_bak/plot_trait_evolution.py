# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:58:32 2015
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
from plot_robot_share import *

run = "V33"

plot_robots = False

#prefix = "./data/" + run + "/" + run + "_micmac_"
prefix = "./data/" + run + "_micmac_"

transform = pickle.load(open(prefix+"st.p", "rb"))
deploy_robots = pickle.load(open(prefix+"dre.p", "rb"))
graph = pickle.load(open(prefix+"graph.p", "rb"))
deploy_traits_init = pickle.load(open(prefix+"dti.p", "rb"))
deploy_traits_desired = pickle.load(open(prefix+"dtd.p", "rb"))

delta_t = 0.04 # time step

# Plot initial and final graphs
fig1 = nxmod.draw_circular(deploy_traits_init, graph, linewidths=3)
plt.axis('equal')
plt.show()
fig2 = nxmod.draw_circular(deploy_traits_desired,graph, linewidths=3)
plt.axis('equal')
plt.show()


# Plot robot distributions.
if plot_robots:
    for index in range(deploy_robots.shape[2]):
        plot_robot_share(deploy_robots, delta_t=delta_t, robot_index=index,
                         cmap_name='Spectral')
        plt.show()

# Plot trait distributions.
for index in range(transform.shape[1]):
    fig = plot_trait_share(deploy_robots, transform=transform, delta_t=delta_t,
                     trait_index=index, cmap_name='Spectral')
    #plt.axes().set_aspect(0.5,'box')
    #plt.axis('equal')
                     
    plt.show()
    fig.savefig('./plots/' + run + '_evol_trait_' + str(index) + '.eps') 