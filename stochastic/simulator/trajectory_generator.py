# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:34:19 2016
@author: amandaprorok

"""


import numpy as np
import scipy as sp
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import networkx as nx
import sys
import time
import pickle
import csv

# my modules
sys.path.append('../plotting')
sys.path.append('../utilities')
sys.path.append('..')

import funcdef_draw_network as nxmod


verbose = False

# -----------------------------------------------------------------------------#
# utility functions

def compute_velocity(x, radius, center, velocity_on_circle):
        
    dx = np.zeros((2, 1))
    tx = x - center
    dx[0] =   tx[0] + tx[1] - tx[0] * (tx[0]**2 + tx[1]**2) / (radius * radius)
    dx[1] = - tx[0] + tx[1] - tx[1] * (tx[0]**2 + tx[1]**2) / (radius * radius)
    dx = dx * velocity_on_circle
  
    return np.squeeze(np.transpose(dx))

def colors_from(cmap_name, ncolors):
    cm = plt.get_cmap(cmap_name)
    cm_norm = matplotlib.colors.Normalize(vmin=0, vmax=ncolors - 1)
    scalar_map = matplotlib.cm.ScalarMappable(norm=cm_norm, cmap=cm)
    return [scalar_map.to_rgba(i) for i in range(ncolors)]

# -----------------------------------------------------------------------------#
# import data

load_data = True
load_run = 'T2'
load_prefix = "../data/" + load_run + "_"

# species-trait matrix
species_traits = pickle.load(open(load_prefix+"species_traits.p", "rb"))

# initial robot distribution
deploy_robots_init = pickle.load(open(load_prefix+"deploy_robots_init.p", "rb"))
deploy_traits_desired = pickle.load(open(load_prefix+"deploy_traits_desired.p", "rb"))

opt_max_rate = pickle.load(open(load_prefix+"max_rate.p", "rb"))
opt_t_max = pickle.load(open(load_prefix+"t_max_sim.p", "rb"))

# transition rates
transition_r = pickle.load(open(load_prefix+"transition_m_init.p", "rb"))  

# graph
graph = pickle.load(open(load_prefix+"graph.p", "rb"))


# -----------------------------------------------------------------------------#
# setup

# constants
velocity_on_circle = 0.15
min_velocity = 0.05
max_velocity = 0.3
task_radius = 0.5
arena_size = 10.
max_rate = 1./60.
dt = 0.2
t_setup = 50.

# adjust values from optimization sim.
t_max = opt_t_max / max_rate + t_setup
T = np.arange(0,t_max,dt)
num_timesteps = np.size(T)
transition_r = transition_r / opt_max_rate * max_rate

num_robots = int(np.ceil(np.sum(deploy_robots_init)))
num_species = np.size(species_traits,0)
num_traits = np.size(species_traits,1)
num_tasks = np.size(deploy_robots_init,0)
sum_species = np.sum(deploy_robots_init,0)
robots_pos_init = np.random.rand(num_robots,2)
robots_task_init = np.zeros((num_robots)).astype(int)


# assign initial tasks to all robots
temp_i = 1
max_temp_i = 0
for si in range(num_species):
    max_temp_i = max_temp_i + sum_species[si];
    for ti in range(num_tasks):
        nb = int(round(deploy_robots_init[ti,si]));
        for nbi in range(nb):
            robots_task_init[temp_i-1] = int(ti);
            temp_i = temp_i + 1;
            # Avoid eventual rounding problems.
            if temp_i > max_temp_i:
                break
            
   
# initialize robots
robots_pos = np.zeros((num_robots, num_timesteps, 2))
robots_task = np.zeros((num_robots, num_timesteps)).astype(int)
for i in range(num_robots):
    robots_pos[i, 0, :] = robots_pos_init[i, :]
    robots_task[i, 0] = robots_task_init[i]
    
# assign species    
sum_species_int = (np.round(sum_species)).astype(int)
robots_species = np.zeros((num_robots,1))
i = 0
for s in range(num_species):
    j = sum_species_int[s]
    robots_species[i:i+j] = s
    i = i+j
robots_species = robots_species.astype(int)

# get transition probabilities (from rate matrix)
transition_p = np.zeros((num_tasks, num_tasks, num_species))
for i in range(num_species):
    transition_p[:,:,i] = sp.linalg.expm(transition_r[:,:,i] * dt)


# setup task sites
task_sites = np.zeros((num_tasks, 2));
if num_tasks == 1:
    # hard code 1st task site
    task_sites[0, :] = np.array([0., 0.])
#    if num_tasks == 4:
#        # % CW from top left: 1-3-4-2
#        o = 0.65
#        c = 1.5
#        task_sites[0, :] = [c-o, c+o];
#        task_sites[1, :] = [c-o, c-o];
#        task_sites[2, :] = [c+o, c+o];
#        task_sites[3, :] = [c+o, c-o];
else:
    for i in range(num_tasks):
        a = (i - 1.) / num_tasks * 2. * np.pi
        task_sites[i, :] = np.array([np.cos(a), np.sin(a)]) * (arena_size * num_robots/20.) # - task_radius - 0.1);
    

# -----------------------------------------------------------------------------#
# main loop

for t in range(1,num_timesteps):
    for r in range(num_robots):
        
        task = robots_task[r,t-1]
        task_center = task_sites[task,:]        
        pos = np.squeeze(robots_pos[r,t-1,:])        
        dx = compute_velocity(pos, task_radius, task_center, velocity_on_circle)        
        
        v = np.linalg.norm(dx);
        if v > max_velocity: dx = dx / v * max_velocity
        elif v < min_velocity: dx = dx / v * min_velocity

        # Update position using Euler integration.
        new_pos = pos + dx * dt
        robots_pos[r, t, :] = new_pos.copy()
        
        # Switch to another task?
        if t > t_setup:
            if num_tasks > 1:
                new_task = np.random.choice(num_tasks, 1, p=transition_p[:,task,np.squeeze(robots_species[r])])
            else:
                new_task = 0;
            if new_task != t and verbose: print 'Robot ', r, 'switched to task', new_task
        else:
            new_task = task
        
        robots_task[r, t] = new_task

# -----------------------------------------------------------------------------#
# plots

plt.figure()

plt.plot(task_sites[:,0],task_sites[:,1],'r+')

col = colors_from('jet', num_robots)
for r in range(num_robots):
    px = robots_pos[r,:,0]
    py =  robots_pos[r,:,1]
    plt.scatter(px, py, c=col[r])
    plt.axis('equal')


# -----------------------------------------------------------------------------#
# save trajectories and species information

# csv: species-traits
filename = './csv/' + load_run + "_species_traits.csv"
a = np.zeros((num_species,num_tasks+1))
a[:,0] = np.arange(num_species)
a[:,1:]= species_traits.copy()
np.savetxt(filename, a, delimiter=",")

# csv: task-sites
filename ='./csv/' + load_run + "_task_sites.csv"
a = np.zeros((num_tasks,4+num_traits)) # task-ID, x, y, radius, t1, t2, ...
a[:,0] = np.arange(num_tasks)
a[:,1:3] = task_sites.copy()
a[:,3] = task_radius 
for t in range(num_tasks):
    a[t,4:] = np.round(deploy_traits_desired[t,:])
np.savetxt(filename, a, delimiter=",")

# csv: trajectories
filename = './csv/' + load_run + "_trajectories.csv"
a = np.zeros((num_timesteps*num_robots,5)) # t, ID, species, x, y
i = 0
for t in range(num_timesteps):
    i = t*num_robots
    j = i+num_robots
    a[i:j,0] = t*dt
    a[i:j,1] = range(num_robots)
    a[i:j,2] = np.squeeze(robots_species)
    a[i:j,3:5] = robots_pos[:,t,:] # x, y
np.savetxt(filename, a, delimiter=",")

 
    