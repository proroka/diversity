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


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
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
# 
  
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
    l1 = plt.plot(x,m_mic, color='green', linewidth=2, label='Microscopic')
    err_ax = np.arange(0,num_tsteps,int(num_tsteps/5))
    #plt.errorbar(x[err_ax],m_mic[err_ax],s_mic[err_ax],fmt='o',markersize=3,color='black')
    
    #l1 = plt.plot(x, pc_a[:,1], color='green', linewidth=2, label='explicit')
    plt.fill_between(x, m_mic+s_mic, m_mic-s_mic, facecolor='green', alpha=0.3)    
    #plt.fill_between(x[err_ax], m_mic[err_ax]+s_mic[err_ax], m_mic[err_ax]-s_mic[err_ax], facecolor='green', alpha=0.3)    

    # plot macro
    l2 = plt.plot(x,diffmac_rat, color='blue', linewidth=2, label='Macroscropic')

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
           
            if match:
                ff = 2.0
            else:
                ff = 1.0
           
            diffmic_rat[t,it] = np.sum(diffmic) / (ff*total_num_traits)   
            diffadp_rat[t,it] = np.sum(diffadp) / (ff*total_num_traits) 
            diffmac_rat[t] = np.sum(diffmac) / (ff*total_num_traits)       
        
        
    x = np.arange(0, num_tsteps) * delta_t

    # plot micro with errorbars
    m_mic = np.mean(diffmic_rat,1)
    s_mic = np.std(diffmic_rat,1)
    #y = smooth(m_mic)
    y = m_mic.copy()    
    l1 = plt.plot(x,y, label='Microscopic')
    err_ax = np.arange(0,num_tsteps,int(num_tsteps/20))
    plt.errorbar(x[err_ax],y[err_ax],s_mic[err_ax],fmt='o',markersize=3,color='black')
    
    # plot adaptive micro with errorbars
    m_mic = np.mean(diffadp_rat,1)
    s_mic = np.std(diffadp_rat,1)
    #y = smooth(m_mic)
    y = m_mic.copy()    
    l2 = plt.plot(x,y, label='Adaptive micro.')
    err_ax = np.arange(0,num_tsteps,int(num_tsteps/20))
    plt.errorbar(x[err_ax],y[err_ax],s_mic[err_ax],fmt='o',markersize=3,color='black')
   
    
    # plot macro
    l3 = plt.plot(x,diffmac_rat, label='Macroscropic')

    ax = plt.gca()
    ax.set_ylim([0.0, 1.0]) 
    plt.axes().set_aspect(num_tsteps*delta_t,'box') 

    # plot legend and labels
    plt.legend(loc='upper right', shadow=False, fontsize='large')     
    plt.xlabel('Time [s]')    
    plt.ylabel('Ratio of misplaced traits')
       
    #plt.show()  
    return fig    
    
# -----------------------------------------------------------------------------#
# get ratio of desired vs actual trait distrib for 1 run

def get_traits_ratio_time(deploy_robots, deploy_traits_desired, transform, match, min_val):
    

    num_tsteps = deploy_robots.shape[1]
    total_num_traits = np.sum(deploy_traits_desired)
    
    for t in range(num_tsteps):
        if match==0:
            traits = np.dot(deploy_robots[:,t,:], transform)
            diff = np.abs(np.minimum(traits - deploy_traits_desired, 0))
        else:
            traits = np.dot(deploy_robots[:,t,:], transform)
            diff = np.abs(traits - deploy_traits_desired)  
    
        ratio = np.sum(diff) / total_num_traits  
        if ratio <= min_val:
            return t
        
    return num_tsteps  

# -----------------------------------------------------------------------------#
# get ratio of desired vs actual trait distrib for 1 run, take into account deviation of robot distribution

def get_traits_ratio_time_strict(deploy_robots, deploy_traits_desired, deploy_robots_desired, transform, match, min_val, slack):
    
    num_tsteps = deploy_robots.shape[1]
    total_num_traits = np.sum(deploy_traits_desired)
    total_num_robots = np.sum(deploy_robots_desired)
    
    reached_robots = False
    for t in range(num_tsteps):
        traits = np.dot(deploy_robots[:,t,:], transform)
        diff = np.abs(traits - deploy_traits_desired)
        diffr = np.abs(deploy_robots[:,t,:] - deploy_robots_desired)
        ratio = np.sum(diff) / total_num_traits
        ratior = np.sum(diffr) / total_num_robots

        if ratior <= (min_val*slack):
            reached_robots = True
        if ratio <= min_val and reached_robots:
            return t
        
    return num_tsteps 

# -----------------------------------------------------------------------------#
# get species traits matrix for 4 species and 4 traits

def get_species_trait_matrix_44(rank):
    
    a = np.zeros((6,4,4))    
    # rank 4
    a[0,:,:] = np.array([[1,0,1,0],[1,0,0,1],[0,1,0,1],[0,0,1,0]])   
    a[1,:,:] = np.array([[1,0,1,0],[1,0,0,1],[0,1,0,1],[0,0,0,1]])
    a[2,:,:] = np.array([[1,0,1,0],[1,0,0,1],[0,1,0,0],[0,0,0,1]])
    
    # rank 3 or less
    a[3,:,:] = np.array([[1,0,1,0],[1,0,0,1],[0,1,0,1],[1,1,1,1]])
    a[4,:,:] = np.array([[0,0,1,1],[1,0,1,1],[0,1,0,0],[1,1,1,1]])
    a[5,:,:] = np.array([[0,0,1,1],[0,1,1,1],[0,0,1,0],[0,1,0,1]])
    
    if rank==3:
        r = np.random.randint(3,6)
        return a[r,:,:]
    else:
        r = np.random.randint(0,3)
        return a[r,:,:]
    
    
# -----------------------------------------------------------------------------#
# convergence time plots

def plot_t_converge(delta_t,t_min_mic, t_min_adp, t_min_mac, t_min_ber):
    
    fig = plt.figure()
    ax = plt.gca()
    N = 6   
    
    
    bp = plt.boxplot([delta_t*t_min_mic, delta_t*t_min_adp, delta_t*t_min_mac, delta_t*t_min_ber],
                     notch=0, sym='+', vert=1, whis=1.5) #,medianprops=medianprops)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    
    off = 1.0
    ymin = delta_t * np.min([t_min_mic, t_min_adp])
    ymax = delta_t * np.max([t_min_mic, t_min_adp])
    ax.set_ylim([0, ymax+off])    
    ax.set_xlim([0.5, N-1.5])

    #plt.legend(loc='upper right', shadow=False, fontsize='large')     
    plt.ylabel('Time [s]')    
    plt.xlabel('Optimization Methods')
    #plt.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    labels = ['Micro', 'Adapt', 'Macro', 'Berman']
    x = [1,2,3,4]    
    plt.xticks(x, labels, rotation='vertical')

    return fig
    

# -----------------------------------------------------------------------------#
# convergence time plots

def plot_t_converge_3(delta_t, t_min_mic, t_min_mac, t_min_ber):
    
    fig = plt.figure()
    ax = plt.gca()
    N = 5   
    
    
    bp = plt.boxplot([delta_t*t_min_mic, delta_t*t_min_mac, delta_t*t_min_ber],
                     notch=0, sym='+', vert=1, whis=1.5) #,medianprops=medianprops)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='+')
    
    #off = 1.0
    #ymin = delta_t * np.min([t_min_mic, t_min_mac, t_min_ber])
    #ymax = delta_t * np.max([t_min_mic, t_min_mac, t_min_ber])
    ax.set_ylim([0, 10.0])    
    ax.set_xlim([0.5, N-1.5])

    #plt.legend(loc='upper right', shadow=False, fontsize='large')     
    plt.ylabel('Time [s]')    
    plt.xlabel('Optimization Methods')
    #plt.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    labels = ['Micro', 'Macro', 'Berman']
    x = [1,2,3]    
    plt.xticks(x, labels, rotation='vertical')

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




# -----------------------------------------------------------------------------#
# plot ratio of desired vs actual robot distribution
# for each hop
  
def plot_traits_ratio_time_mic_distributed(deploy_robots_mic_hop, deploy_robots_mac, deploy_traits_desired, transform, delta_t, match):
    # deploy_robots_mic_hop[nodes,ts,S,it,nh]
    
    fig = plt.figure()
    colors = ['green', 'blue', 'red', 'cyan', 'magenta', 'yellow']    
    
    num_hops = deploy_robots_mic_hop.shape[4]    
    num_tsteps = deploy_robots_mic_hop.shape[1]
    total_num_traits = np.sum(deploy_traits_desired)    
    num_it = deploy_robots_mic_hop.shape[3]
        
    for nh in range(num_hops):   
        #deploy_robots_mic = np.mean(deploy_robots_mic_hop[:,:,:,:,nh],3)
        
        
        diffmic_rat = np.zeros((num_tsteps, num_it))
        diffmac_rat = np.zeros(num_tsteps)
    
        for it in range(num_it):
            deploy_robots_mic = deploy_robots_mic_hop[:,:,:,it,nh]
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
                
                diffmic_rat[t,it] = np.sum(diffmic) / (2*total_num_traits)      
                diffmac_rat[t] = np.sum(diffmac) / (2*total_num_traits)       
            
            
        x = np.arange(0, num_tsteps) * delta_t
    
        # plot micro with errorbars
        m_mic = np.mean(diffmic_rat,1)
        s_mic = np.std(diffmic_rat,1)
        
        lab = 'H-'+str(nh)
        col = colors[nh]
        l1 = plt.plot(x,m_mic, color=col, linewidth=2, label=lab)
        err_ax = np.arange(0,num_tsteps,int(num_tsteps/5))
        #plt.errorbar(x[err_ax],m_mic[err_ax],s_mic[err_ax],fmt='o',markersize=3,color='black')
        
        #l1 = plt.plot(x, pc_a[:,1], color='green', linewidth=2, label='explicit')
        plt.fill_between(x, m_mic+s_mic, m_mic-s_mic, facecolor=col, alpha=0.3)    
        #plt.fill_between(x[err_ax], m_mic[err_ax]+s_mic[err_ax], m_mic[err_ax]-s_mic[err_ax], facecolor='green', alpha=0.3)    
    
        # plot macro
        #l2 = plt.plot(x,diffmac_rat, color='blue', linewidth=2, label='Macroscropic')





    # plot legend and labels
    plt.legend(loc='upper right', shadow=False, fontsize='x-large')     
    plt.xlabel('Time [s]')    
    plt.ylabel('Ratio of misplaced traits')
    

    #plt.show()  
    return fig

# -----------------------------------------------------------------------------#
# get ratio (without plotting)
  
def get_traits_ratio_time_mic(deploy_robots_mic_all, deploy_traits_desired, transform, match):
    # deploy_robots_mic[nodes,ts,S,it]
    
    num_tsteps = deploy_robots_mic_all.shape[1]
    total_num_traits = np.sum(deploy_traits_desired)    
    num_it = deploy_robots_mic_all.shape[3]
        
    diffmic_rat = np.zeros((num_tsteps, num_it))   
    for it in range(num_it):
        deploy_robots_mic = deploy_robots_mic_all[:,:,:,it]
        for t in range(num_tsteps):
            
            if match==0:
                traits = np.dot(deploy_robots_mic[:,t,:], transform)
                diffmic = np.abs(np.minimum(traits - deploy_traits_desired, 0))
                
            else:
                traits = np.dot(deploy_robots_mic[:,t,:], transform)
                diffmic = np.abs(traits - deploy_traits_desired)   
              
            diffmic_rat[t,it] = np.sum(diffmic) / (2*total_num_traits)      
        
    return diffmic_rat

# -----------------------------------------------------------------------------#
# get ratio (without plotting)
  
def get_traits_ratio_time_mic_distributed(deploy_robots_mic_hop, deploy_traits_desired, transform, match):
    # deploy_robots_mic_hop[nodes,ts,S,it,nh]
    
       
    num_hops = deploy_robots_mic_hop.shape[4]    
    num_tsteps = deploy_robots_mic_hop.shape[1]
    total_num_traits = np.sum(deploy_traits_desired)    
    num_it = deploy_robots_mic_hop.shape[3]
        
    diffmic_rat = np.zeros((num_tsteps, num_it, num_hops))   
    for nh in range(num_hops):   
        for it in range(num_it):
            deploy_robots_mic = deploy_robots_mic_hop[:,:,:,it,nh]
            for t in range(num_tsteps):
                
                if match==0:
                    traits = np.dot(deploy_robots_mic[:,t,:], transform)
                    diffmic = np.abs(np.minimum(traits - deploy_traits_desired, 0))
                    
                else:
                    traits = np.dot(deploy_robots_mic[:,t,:], transform)
                    diffmic = np.abs(traits - deploy_traits_desired)   
                  
                diffmic_rat[t,it,nh] = np.sum(diffmic) / (2*total_num_traits)      
            
    return diffmic_rat

    
# -----------------------------------------------------------------------------#
# plot ratio (with input of ratios for all hops)
  
def plot_traits_ratio_time_mic_distributed_multirun(diffmic_ratio, delta_t):
    
    num_hops = diffmic_ratio.shape[2]    
    num_tsteps = diffmic_ratio.shape[0]  
        
    fig = plt.figure()
    colors = ['green', 'blue', 'cyan', 'magenta', 'red', 'yellow']    
    
    for nh in range(num_hops):   
        x = np.arange(0, num_tsteps) * delta_t
    
        # plot micro with errorbars
        m_mic = np.mean(diffmic_ratio[:,:,nh],1)
        s_mic = np.std(diffmic_ratio[:,:,nh],1)
        
        lab = 'H-'+str(nh)
        col = colors[nh]
        l1 = plt.plot(x,m_mic, color=col, linewidth=2, label=lab)

        err_ax = np.arange(0,num_tsteps,int(num_tsteps/10))+nh
        
        plt.errorbar(x[err_ax],m_mic[err_ax],s_mic[err_ax],fmt='o',markersize=3,color=col)

        #plt.fill_between(x, m_mic+s_mic, m_mic-s_mic, facecolor=col, alpha=0.3)    


    # plot legend and labels
    plt.legend(loc='upper right', shadow=False, fontsize='x-large')     
    plt.xlabel('Time [s]')    
    plt.ylabel('Ratio of misplaced traits')
        
    #0plt.gca().set_aspect()
    #plt.axis('equal') 
    plt.axes().set_aspect(7.0,'box')    
    
    #plt.show()  
    return fig

    
# -----------------------------------------------------------------------------#
# plot ratio (with input of ratios for all hops)
  
def plot_traits_ratio_time_micmic_multirun(diffmic_ratio, diffmic_adp_ratio, delta_t):
    
    num_tsteps = diffmic_ratio.shape[0]  
        
    fig = plt.figure()
    colors = ['green', 'blue', 'red', 'cyan', 'magenta', 'yellow']    
    
     
    x = np.arange(0, num_tsteps) * delta_t

    # plot micro with errorbars
    m_mic = np.mean(diffmic_ratio[:,:],1)
    s_mic = np.std(diffmic_ratio[:,:],1)

    m_mic_adp = np.mean(diffmic_adp_ratio[:,:],1)
    s_mic_adp = np.std(diffmic_adp_ratio[:,:],1)    
    
    
    col = 'blue'
    l1 = plt.plot(x,m_mic, color=col, linewidth=2, label='Micro')
    err_ax = np.arange(0,num_tsteps,int(num_tsteps/5))
    plt.fill_between(x, m_mic+s_mic, m_mic-s_mic, facecolor=col, alpha=0.3)    

    col = 'green'
    l2 = plt.plot(x,m_mic_adp, color=col, linewidth=2, label='Micro Adapt.')
    err_ax = np.arange(0,num_tsteps,int(num_tsteps/5))
    plt.fill_between(x, m_mic_adp+s_mic_adp, m_mic_adp-s_mic_adp, facecolor=col, alpha=0.3)    


    # plot legend and labels
    plt.legend(loc='upper right', shadow=False, fontsize='x-large')     
    plt.xlabel('Time [s]')    
    plt.ylabel('Ratio of misplaced traits')
        
    
    #plt.show()  
    return fig    
    
    
    
    
    
    
    
    
    
    
    
    