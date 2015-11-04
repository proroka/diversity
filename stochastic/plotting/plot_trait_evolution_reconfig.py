# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:39:06 2015

@author: amanda
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

# V26 used for ICRA 2016
run = "V26"

plot_robots = False

prefix = "./data/" + run + "/" + run + "_micmac_"

transform = pickle.load(open(prefix+"st.p", "rb"))
deploy_robots = pickle.load(open(prefix+"drev.p", "rb"))
graph = pickle.load(open(prefix+"graph.p", "rb"))

deploy_traits_init_0 = pickle.load(open(prefix+"ti_0.p", "rb"))
deploy_traits_desired_0 = pickle.load(open(prefix+"td_0.p", "rb"))

deploy_traits_init_1 = pickle.load(open(prefix+"ti_1.p", "rb"))
deploy_traits_desired_1 = pickle.load(open(prefix+"td_1.p", "rb"))

deploy_traits_init_2 = pickle.load(open(prefix+"ti_2.p", "rb"))
deploy_traits_desired_2 = pickle.load(open(prefix+"td_2.p", "rb"))

delta_t = 0.04 # time step

# -----------------------------------------------------------------------------#
# Plot initial and final graphs

fig1 = nxmod.draw_circular(deploy_traits_init_0, graph, linewidths=3)
plt.axis('equal')
plt.show()
fig2 = nxmod.draw_circular(deploy_traits_desired_0,graph, linewidths=3)
plt.axis('equal')
plt.show()

#fig3 = nxmod.draw_circular(deploy_traits_init_1, graph, linewidths=3)
#plt.axis('equal')
#plt.show()
fig4 = nxmod.draw_circular(deploy_traits_desired_1,graph, linewidths=3)
plt.axis('equal')
plt.show()

#fig5 = nxmod.draw_circular(deploy_traits_init_2, graph, linewidths=3)
#plt.axis('equal')
#plt.show()
fig6 = nxmod.draw_circular(deploy_traits_desired_2,graph, linewidths=3)
plt.axis('equal')
plt.show()

# -----------------------------------------------------------------------------#
# plot flows

#  robot distributions.
if plot_robots:
    for index in range(deploy_robots.shape[2]):
        plot_robot_share(deploy_robots, delta_t=delta_t, robot_index=index,
                         cmap_name='Spectral')
        plt.show()

#  trait distributions.
for index in range(transform.shape[1]):
    figE = plot_trait_share(deploy_robots, transform=transform, delta_t=delta_t,
                     trait_index=index, cmap_name='Spectral')

    plt.show()
    figE.savefig('./plots/' + run + '_evol_trait_' + str(index) + '.eps') 
    
# -----------------------------------------------------------------------------#
# save figs   
 
fig1.savefig('./plots/' + run + '_gi_0.eps') 
fig2.savefig('./plots/' + run + '_gd_0.eps') 

#fig3.savefig('./plots/' + run + '_gi_1.eps') 
fig4.savefig('./plots/' + run + '_gd_1.eps') 

#fig5.savefig('./plots/' + run + '_gi_2.eps') 
fig6.savefig('./plots/' + run + '_gd_2.eps') 



   
    