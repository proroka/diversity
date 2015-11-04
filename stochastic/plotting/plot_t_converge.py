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


run = 'V12'
save_plots = False

# -----------------------------------------------------------------------------#
def plot_t_converge(delta_t,t_min_mic, t_min_adp, t_min_ber):
    
    fig = plt.figure()
    ax = plt.gca()
    N = 5  
    
    
    bp = plt.boxplot([delta_t*t_min_ber, delta_t*t_min_mic, delta_t*t_min_adp],
                     notch=0, sym='+', vert=1, whis=1.5) #,medianprops=medianprops)
                     
                     
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.grid(axis='y')
    off = 1.0
    ymin = delta_t * np.min([t_min_mic, t_min_adp, t_min_ber])
    ymax = delta_t * np.max([t_min_mic, t_min_adp, t_min_ber])
    ax.set_ylim([0, ymax+off])    
    ax.set_xlim([0.5, N-1.5])

    # print values
    med_ber=  np.median(delta_t*t_min_ber)    
    med_mic=  np.median(delta_t*t_min_mic)
    med_adp=  np.median(delta_t*t_min_adp)
    print ' *** '
    print 'Median ber: ', med_ber
    print 'Median mic: ', med_mic
    print 'Median adp: ', med_adp
    imp = (med_ber-med_mic)/med_ber
    imp2 = (med_mic-med_adp)/med_mic
    print 'Impr. of mic over ber: ', imp
    print 'Impr. of adp over mic: ', imp2

    #plt.legend(loc='upper right', shadow=False, fontsize='large')     
    plt.ylabel('T')    
    #plt.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    labels = ['a', 'b', 'c']
    x = [1,2,3]    
    plt.xticks(x, labels) #, rotation='vertical')

    return fig
    
# -----------------------------------------------------------------------------#
def plot_t_converge_mod(delta_t,t_min_mic, t_min_ber):
    
    fig = plt.figure()
    ax = plt.gca()
    N = 4  
    
    
    bp = plt.boxplot([delta_t*t_min_ber, delta_t*t_min_mic],
                     notch=0, sym='+', vert=1, whis=1.5) #,medianprops=medianprops)
                     
                     
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    plt.grid(axis='y')
    off = 1.0
    ymin = delta_t * np.min([t_min_mic, t_min_ber])
    ymax = delta_t * np.max([t_min_mic, t_min_ber])
    ax.set_ylim([0, ymax+off])    
    ax.set_xlim([0.5, N-1.5])

    # print values
    med_ber=  np.median(delta_t*t_min_ber)    
    med_mic=  np.median(delta_t*t_min_mic)
    print ' *** '
    print 'Median ber: ', med_ber
    print 'Median mic: ', med_mic
    imp = (med_ber-med_mic)/med_ber
    print 'Impr. of mic over ber: ', imp

    #plt.legend(loc='upper right', shadow=False, fontsize='large')     
    plt.ylabel('T')    
    #plt.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    labels = ['a', 'b']
    x = [1,2]    
    plt.xticks(x, labels) #, rotation='vertical')

    return fig
    
# -----------------------------------------------------------------------------#
# load data

prefix = "./data/" + run + "/" + run + "_"
#prefix = "./data/" + run + "_"

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
r = 3
tmm = t_min_mic[ranks==r]
tma = t_min_adp[ranks==r]
tmb = t_min_mic_ber[ranks==r]
fig1 = plot_t_converge(delta_t, tmm, tma, tmb)
plt.axes().set_aspect(0.5,'box')


r = 4
tmm = t_min_mic[ranks==r]
tma = t_min_adp[ranks==r]
tmb = t_min_mic_ber[ranks==r]
fig2 = plot_t_converge(delta_t, tmm, tma, tmb)
plt.axes().set_aspect(0.5,'box')

fig3 = plot_t_converge(delta_t, t_min_mic, t_min_adp, t_min_mic_ber)

fig4 = plot_t_converge_mod(delta_t, t_min_mic, t_min_mic_ber)
    
#plt.show()


# -----------------------------------------------------------------------------#
# save plots
 
if save_plots:
    fig1.savefig(prefix+'rank3_t_conv.eps') 
    fig2.savefig(prefix+'rank4_t_conv.eps')                      
    fig3.savefig(prefix+'all_t_conv.eps')  
    fig4.savefig(prefix+'short_t_conv.eps')                         



