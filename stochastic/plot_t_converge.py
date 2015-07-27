# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:40:07 2015
@author: amandaprorok

"""



import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import pickle

# my modules
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod


run = 'V13'

# -----------------------------------------------------------------------------#
def plot_t_converge(delta_t,t_min_mic, t_min_adp, t_min_ber):
    
    fig = plt.figure()
    ax = plt.gca()
    N = 5  
    
    
    bp = plt.boxplot([delta_t*t_min_mic, delta_t*t_min_adp, delta_t*t_min_ber],
                     notch=0, sym='+', vert=1, whis=1.5) #,medianprops=medianprops)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    
    off = 1.0
    ymin = delta_t * np.min([t_min_mic, t_min_adp, t_min_ber])
    ymax = delta_t * np.max([t_min_mic, t_min_adp, t_min_ber])
    ax.set_ylim([0, ymax+off])    
    ax.set_xlim([0.5, N-1.5])

    #plt.legend(loc='upper right', shadow=False, fontsize='large')     
    plt.ylabel('Time [s]')    
    plt.xlabel('Optimization Methods')
    #plt.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    labels = ['J3', 'J3-tilde', 'Convex']
    x = [1,2,3]    
    plt.xticks(x, labels) #, rotation='vertical')

    return fig
    

# -----------------------------------------------------------------------------#
# load data

#prefix = "./data/" + run + "/" + run + "_"
prefix = "./data/" + run + "_"

rank_Q = pickle.load(open(prefix+"rank_Q.p", "rb"))
t_min_mic = pickle.load(open(prefix+"t_min_mic.p", "rb"))
t_min_mic = pickle.load(open(prefix+"t_min_mic.p", "rb"))
t_min_mac = pickle.load(open(prefix+"t_min_mac.p", "rb"))
t_min_adp = pickle.load(open(prefix+"t_min_adp.p", "rb"))
t_min_mic_ber = pickle.load(open(prefix+"t_min_mic_ber.p", "rb"))
#t_min_mac_ber = pickle.load(open(prefix+"t_min_mac_ber.p", "rb"))

# -----------------------------------------------------------------------------#
# split data by rank

num_iter = t_min_adp.shape[0]
num_g_iter = t_min_mac.shape[0]
npi = num_iter / num_g_iter
ranks = np.zeros(num_iter)
for i in range(num_g_iter):
    ranks[i*npi:(i+1)*npi] = np.repeat(rank_Q[i],npi)

# -----------------------------------------------------------------------------#
# plot
delta_t = 0.04
# plot traits ratio
#fig3 = plot_traits_ratio_time_micmicmac(deploy_robots_micro, deploy_robots_micro_adapt, deploy_robots_euler, deploy_traits_desired,species_traits, delta_t, match)

# plot time at which min ratio reached
tmm = t_min_mic[ranks==3]
tma = t_min_adp[ranks==3]
tmb = t_min_mic_ber[ranks==3]
fig1 = plot_t_converge(delta_t, tmm, tma, tmb)

tmm = t_min_mic[ranks==4]
tma = t_min_adp[ranks==4]
tmb = t_min_mic_ber[ranks==4]
fig2 = plot_t_converge(delta_t, tmm, tma, tmb)

fig3 = plot_t_converge(delta_t, t_min_mic, t_min_adp, t_min_mic_ber)

#plt.show()


# -----------------------------------------------------------------------------#
# save plots
 
save_plots = True
if save_plots:
    fig1.savefig(prefix+'rank3_t_conv.eps') 
    fig2.savefig(prefix+'rank4_t_conv.eps')                      
    fig3.savefig(prefix+'all_t_conv.eps')                      



