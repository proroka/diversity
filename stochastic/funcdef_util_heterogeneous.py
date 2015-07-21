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
# from wiki.scipy.org/Cookbook/SignalSmooth

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    #return y
    return y[(window_len/2-1):-(window_len/2)-1]



# -----------------------------------------------------------------------------#
# plot ratio of desired vs actual robot distribution

def plot_robots_ratio_time_micmac(deploy_robots_mic, deploy_robots_mac, deploy_robots_desired, delta_t):
    plot_option = 0 # 0: ratio, 1: cost
    num_iter = deploy_robots_mic.shape[1]
    total_num_robots = np.sum(deploy_robots_mic[:,0,:])
    
    diffmic_sqs = np.zeros(num_iter)
    diffmac_sqs = np.zeros(num_iter)
    diffmic_rat = np.zeros(num_iter)
    diffmac_rat = np.zeros(num_iter)
    for t in range(num_iter):
        diffmic = np.abs(deploy_robots_mic[:,t,:] - deploy_robots_desired)    
        diffmac = np.abs(deploy_robots_mac[:,t,:] - deploy_robots_desired) 
        diffmic_rat[t] = np.sum(diffmic) / total_num_robots       
        diffmic_sqs[t] = np.sum(np.square(diffmic))
        diffmac_rat[t] = np.sum(diffmac) / total_num_robots 
        diffmac_sqs[t] = np.sum(np.square(diffmac))
        
    x = np.arange(0, num_iter) * delta_t
    if(plot_option==0):
        l1 = plt.plot(x,diffmic_rat)
        l2 = plt.plot(x,diffmac_rat)
    if(plot_option==1):
        l1 = plt.plot(x,diffmic_sqs)
        l2 = plt.plot(x,diffmac_sqs)
    
    plt.xlabel('time [s]')    
    plt.ylabel('ratio of misplaced robots')
    plt.legend((l1, l2),('Micro','Macro'))
    plt.show()
    
# -----------------------------------------------------------------------------#
# plot ratio of desired vs actual robot distribution

def plot_traits_ratio_time_micmac(deploy_robots_micro, deploy_robots_mac, deploy_traits_desired, transform, delta_t, match):
    
    fig = plt.figure()
    
    deploy_robots_mic = np.mean(deploy_robots_micro,3)

    plot_option = 0 # 0: ratio, 1: cost
    num_tsteps = deploy_robots_mic.shape[1]
    total_num_traits = np.sum(deploy_traits_desired)

    #deploy_robots_micro[:,:,:,it]
    num_it = deploy_robots_micro.shape[3]
    diffmic_rat = np.zeros((num_tsteps, num_it))
    diffmac_rat = np.zeros(num_tsteps)

    for it in range(num_it):
        deploy_robots_mic = deploy_robots_micro[:,:,:,it]
        for t in range(num_tsteps):
            
            if match==0:
                traits = np.dot(deploy_robots_mic[:,t,:], transform)
                diffmic = np.abs(np.minimum(traits - deploy_traits_desired, 0))
                traits = np.dot(deploy_robots_mac[:,t,:], transform)
                diffmac = np.abs(np.minimum(traits - deploy_traits_desired, 0))
            else:
                traits = np.dot(deploy_robots_mic[:,t,:], transform)
                diffmic = np.abs(traits - deploy_traits_desired)   
                traits = np.dot(deploy_robots_mac[:,t,:], transform)
                diffmac = np.abs(traits - deploy_traits_desired)  
            
            diffmic_rat[t,it] = np.sum(diffmic) / total_num_traits      
            diffmac_rat[t] = np.sum(diffmac) / total_num_traits       
        
        
    x = np.arange(0, num_tsteps) * delta_t

    # plot micro with errorbars
    m_mic = np.mean(diffmic_rat,1)
    s_mic = np.std(diffmic_rat,1)
    l1 = plt.plot(x,m_mic, label='Microscopic')
    err_ax = np.arange(0,num_tsteps,int(num_tsteps/20))
    plt.errorbar(x[err_ax],m_mic[err_ax],s_mic[err_ax],fmt='o',markersize=3,color='black')
    
    # plot macro
    l2 = plt.plot(x,diffmac_rat, label='Macroscropic')

    # plot legend and labels
    plt.legend(loc='upper right', shadow=False, fontsize='x-large')     
    plt.xlabel('Time [s]')    
    plt.ylabel('Ratio of misplaced traits')
       
    #plt.show()  
    return fig

# -----------------------------------------------------------------------------#
# plot ratio of desired vs actual robot distribution

def plot_traits_ratio_time_micmicmac(deploy_robots_micro, deploy_robots_micro_adapt, deploy_robots_mac, 
                                     deploy_traits_desired, transform, delta_t, match):
    
    fig = plt.figure()
    
    deploy_robots_mic = np.mean(deploy_robots_micro,3)
    deploy_robots_adp = np.mean(deploy_robots_micro_adapt,3)
    
    num_tsteps = deploy_robots_mic.shape[1]
    total_num_traits = np.sum(deploy_traits_desired)

    #deploy_robots_micro[:,:,:,it]
    num_it = deploy_robots_micro.shape[3]
    diffmic_rat = np.zeros((num_tsteps, num_it))
    diffadp_rat = np.zeros((num_tsteps, num_it))
    diffmac_rat = np.zeros(num_tsteps)

    for it in range(num_it):
        deploy_robots_mic = deploy_robots_micro[:,:,:,it]
        deploy_robots_adp = deploy_robots_micro_adapt[:,:,:,it]
        for t in range(num_tsteps):
            
            if match==0:
                traits = np.dot(deploy_robots_mic[:,t,:], transform)
                diffmic = np.abs(np.minimum(traits - deploy_traits_desired, 0))
                
                traits = np.dot(deploy_robots_adp[:,t,:], transform)
                diffadp = np.abs(np.minimum(traits - deploy_traits_desired, 0))
                
                traits = np.dot(deploy_robots_mac[:,t,:], transform)
                diffmac = np.abs(np.minimum(traits - deploy_traits_desired, 0))
            else:
                traits = np.dot(deploy_robots_mic[:,t,:], transform)
                diffmic = np.abs(traits - deploy_traits_desired)   
                
                traits = np.dot(deploy_robots_adp[:,t,:], transform)
                diffadp = np.abs(traits - deploy_traits_desired)   
                
                traits = np.dot(deploy_robots_mac[:,t,:], transform)
                diffmac = np.abs(traits - deploy_traits_desired)  
            
            diffmic_rat[t,it] = np.sum(diffmic) / total_num_traits   
            diffadp_rat[t,it] = np.sum(diffadp) / total_num_traits 
            diffmac_rat[t] = np.sum(diffmac) / total_num_traits       
        
        
    x = np.arange(0, num_tsteps) * delta_t

    # plot micro with errorbars
    m_mic = np.mean(diffmic_rat,1)
    s_mic = np.std(diffmic_rat,1)
    y = smooth(m_mic)
    l1 = plt.plot(x,y, label='Microscopic')
    err_ax = np.arange(0,num_tsteps,int(num_tsteps/20))
    plt.errorbar(x[err_ax],y[err_ax],s_mic[err_ax],fmt='o',markersize=3,color='black')
    
    # plot adaptive micro with errorbars
    m_mic = np.mean(diffadp_rat,1)
    s_mic = np.std(diffadp_rat,1)
    y = smooth(m_mic)
    l2 = plt.plot(x,y, label='Adaptive micro.')
    err_ax = np.arange(0,num_tsteps,int(num_tsteps/20))
    plt.errorbar(x[err_ax],y[err_ax],s_mic[err_ax],fmt='o',markersize=3,color='black')
    
    # plot macro
    l3 = plt.plot(x,diffmac_rat, label='Macroscropic')

    # plot legend and labels
    plt.legend(loc='upper right', shadow=False, fontsize='large')     
    plt.xlabel('Time [s]')    
    plt.ylabel('Ratio of misplaced traits')
       
    #plt.show()  
    return fig    
    
    
# -----------------------------------------------------------------------------#
# plot ratio of desired vs actual robot distribution

def plot_robots_cost_time(deploy_robots, deploy_robots_desired):
    
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
    
def plot_traits_cost_time(deploy_robots, deploy_traits_desired, transform):
    
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
    # draw graph with node size proportional to trait distribution
    
    nx.draw_circular(graph)
    
    print "initial config:"
    scale = 1000 / np.mean(np.sum(deploy_traits_init,1)) # scale size of node
    nx.draw_circular(graph, node_size=np.sum(deploy_traits_init,1)*scale)
    plt.show()
    
    print "final config:"
    scale = 1000 / np.mean(np.sum(deploy_traits_final,1)) # scale size of node
    nx.draw_circular(graph, node_size=np.sum(deploy_traits_final,1)*scale)
    plt.show()
    
    #nx.draw(graph)
    #dg = nx.DiGraph(graph)
    #nx.draw(dg)

    #nx.draw_circular(graph, node_size=np.sum(deploy_traits_init,1)*scale)
    #nx. draw_networkx_labels(G, pos[, labels, ...])

    # draw graph with node size proportional to robot population
    # print 'Final configuration:'
    # nx.draw_circular(graph, node_size=deploy_robots[:,t_max-1,s_ind]*scale)
    # nx.draw_circular(graph, node_size=np.sum(deploy_traits_final,1)*scale)
    # plt.show()



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

def plot_robots_time_micmac(deploy_robots_mic, deploy_robots_mac, species_ind, node_ind):
    
    fig = plt.figure()
    #plt.axis('equal')
    num_nodes = deploy_robots_mic.shape[0]
    num_iter = deploy_robots_mic.shape[1]

    delta_t = 0.04
    x = np.arange(0, num_iter) * delta_t


    # plot evolution of robot population over time
    labeled = False
    for n in node_ind:    
        y = deploy_robots_mic[n,:,species_ind]
        y2 = deploy_robots_mac[n,:,species_ind]

        if (not labeled):
            labeled = True
            plt.plot(x,y,color='green', label='Microscopic')
            plt.plot(x,y2, color='red', label='Macroscopic')
        else:    
            plt.plot(x,y,color='green')
            plt.plot(x,y2, color='red')
            
    # plot legend and labels
    plt.legend(loc='upper right', shadow=False, fontsize='large')     
    plt.xlabel('Time [s]')    
    plt.ylabel('Number of robots')
    
    #plt.show()
    return fig
    
# -----------------------------------------------------------------------------#
# plot
def plot_traits_time_micmac(deploy_robots_mic, deploy_robots_mac, species_traits, node_ind, trait_ind):
    fig = plt.figure()
     
    num_nodes = deploy_robots_mic.shape[0]
    num_iter = deploy_robots_mic.shape[1]
    num_traits = species_traits.shape[1]
    
    delta_t = 0.04
    x = np.arange(0, num_iter) * delta_t
    
    deploy_traits_mic = np.zeros((num_nodes, num_iter, num_traits))
    deploy_traits_mac = np.zeros((num_nodes, num_iter, num_traits))
    for t in range(num_iter):
        deploy_traits_mic[:,t,:] = np.dot(deploy_robots_mic[:,t,:], species_traits)
        deploy_traits_mac[:,t,:] = np.dot(deploy_robots_mac[:,t,:], species_traits)
        
    # plot evolution of robot population over time
    labeled = False
    for n in node_ind:    
        y = deploy_traits_mic[n,:,trait_ind]
        y2 = deploy_traits_mac[n,:,trait_ind]

        if (not labeled):
            labeled = True
            plt.plot(x,y,color='green', label='Microscopic')
            plt.plot(x,y2, color='red', label='Macroscopic')
        else:    
            plt.plot(x,y,color='green')
            plt.plot(x,y2, color='red')
            
    # plot legend and labels
    plt.legend(loc='upper right', shadow=False, fontsize='large')     
    plt.xlabel('Time [s]')    
    plt.ylabel('Number of traits')
    #plt.gca().set_axis('equal', adjustable='box')
    #plt.axis('equal')    
    #plt.show() 
    return fig
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    