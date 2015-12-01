# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:13:22 2015
@author: amandaprorok

"""

import numpy as np
import scipy as sp
import scipy.io
import scipy.ndimage.filters
import pylab as pl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import sys
import pickle
import time

# my modules
sys.path.append('../../plotting')
sys.path.append('../../utilities')
sys.path.append('../..')

from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod


##-------------------------------

# Extra control vars.
save_movie = False
save_fig = True
movie_filename = 'run.mp4'
show_movie = False
show_plots = True
remove_setup_time = True
save_data = False
use_error_bars = False


##-------------------------------

# Set these variables as in optimize_for_boats.py
match = 1
l_norm = 2

nspecies = 3
ntraits = 4

# Choose 1 out of 3 segments for evolution plot
segment = 1

# Matlab run files
# Specify multiple runs if desired.
mat_run = [501, 502, 503, 504, 506, 507]

#---------------------------------------------------

# Version of code for TRO

run = "B01"

plot_robots = False

prefix = "../../data/" + run + "/" + run + "_evolution_"

species_traits = pickle.load(open(prefix+"st.p", "rb"))

# this contains the 3 time segments
deploy_robots = pickle.load(open(prefix+"drev.p", "rb"))

graph = pickle.load(open(prefix+"graph.p", "rb"))

deploy_traits_init_0 = pickle.load(open(prefix+"ti_0.p", "rb"))
deploy_traits_desired_0 = pickle.load(open(prefix+"td_0.p", "rb"))

deploy_traits_init_1 = pickle.load(open(prefix+"ti_1.p", "rb"))
deploy_traits_desired_1 = pickle.load(open(prefix+"td_1.p", "rb"))

deploy_traits_init_2 = pickle.load(open(prefix+"ti_2.p", "rb"))
deploy_traits_desired_2 = pickle.load(open(prefix+"td_2.p", "rb"))


#---------------------------------------------------
# Set the initial robot distribution used.


time_steps = deploy_robots.shape[1]
ind = time_steps / 3

deploy_robots_init_0 = deploy_robots[:,0,:]
deploy_robots_final_0 = deploy_robots[:,ind,:]

deploy_robots_init_1 = deploy_robots[:,ind,:]
deploy_robots_final_1 = deploy_robots[:,2*ind-1,:]

deploy_robots_init_2= deploy_robots[:,2*ind,:]
deploy_robots_final_2 = deploy_robots[:,3*ind-1,:]

# Open first run for getting most variables.
matlab_workspace_file = '../matlab/data/run_' + str(mat_run[0]) + '_all_data.mat'


####################
# Rest of the code #
####################

# Load mat file and get relevant matrices.
matlab_data = sp.io.loadmat(matlab_workspace_file)

task_sites = matlab_data['task_sites']
ntasks = task_sites.shape[0]
nslots = matlab_data['nslots']
auto_advance = matlab_data['auto_advance']

boats_species = np.squeeze(matlab_data['boats_species']) - 1
nboats = boats_species.shape[0]
delta_t = matlab_data['dt'][0][0]
setup_time = matlab_data['setup_time']
t_max = matlab_data['max_time'] - (setup_time if remove_setup_time else 0)

# scale number of boats, multiple of 100
f = nboats / 100.0

if auto_advance:
    segment == 1
    switching_steps = matlab_data['switching_steps'][0].astype(int) - 1
    if remove_setup_time:
        switching_steps -= round(setup_time / delta_t)

deploy_robots_init = [
    f * deploy_robots_init_0,
    f * deploy_robots_init_1,
    f * deploy_robots_init_2,
]
deploy_robots_final = [
    f * deploy_robots_final_0,
    f * deploy_robots_final_1,
    f * deploy_robots_final_2,
]

print 'Number of robots =', np.sum(deploy_robots_init[0])
print 'Number of robots per species =', np.sum(deploy_robots_init[0], axis=0)

# Get trait distributions.
deploy_traits_init = []
deploy_traits_desired = []
for di, df in zip(deploy_robots_init, deploy_robots_final):
    deploy_traits_init.append(np.dot(di, species_traits))
    deploy_traits_desired.append(np.dot(df, species_traits))

# Build K as in the python code.
Ks = []
for i, ks in enumerate(matlab_data['K']):
    Ks.append(np.empty((ntasks, ntasks, nspecies)))
    for s, k in enumerate(ks[0]):
        Ks[-1][:, :, s] = k[0]

arena_size = matlab_data['arena_size'][0][0]
task_radius = matlab_data['task_radius']

deploy_boats = []
for r in mat_run:
    matlab_workspace_file = '../matlab/data/run_' + str(r) + '_all_data.mat'
    matlab_data = sp.io.loadmat(matlab_workspace_file)
    boats_pos = matlab_data['boats_pos']
    print 'Number of time-steps =', boats_pos.shape[1]

    # Remove setup-time.
    if remove_setup_time:
        boats_pos = boats_pos[:, int(setup_time / delta_t):, :]
    # Smooth boats positions.
    valid_timesteps = boats_pos.shape[1]
    for i in range(nboats):
        boats_pos[i, :, 0] = sp.ndimage.filters.median_filter(boats_pos[i, :, 0], size=10, mode='reflect')
        boats_pos[i, :, 1] = sp.ndimage.filters.median_filter(boats_pos[i, :, 1], size=10, mode='reflect')
    # Compute per timesteps the number of boats in each task.
    deploy_boats.append(np.zeros((ntasks, valid_timesteps, nspecies)))
    ntimesteps = boats_pos.shape[1]
    print 'Number of actual time-steps =', ntimesteps

    distance_threshold = 0.1
    # Get initial task as the closest one.
    closest_task = []
    t = 0
    positions = boats_pos[:, t, :]
    for b in range(nboats):
        position = positions[b, :]
        closest_task.append(np.argmin(np.sum(np.square(task_sites - position), axis=1)))
        species = boats_species[b]
        deploy_boats[-1][closest_task[b], t, species] += 1
    # Only switch when distance to new task is small enough.
    for t in range(1, ntimesteps):
        positions = boats_pos[:, t, :]
        for b in range(nboats):
            position = positions[b, :]
            distance_to_closest = np.sqrt(np.min((np.sum(np.square(task_sites - position), axis=1))))
            if distance_to_closest < distance_threshold:
                closest_task[b] = np.argmin(np.sum(np.square(task_sites - position), axis=1))
            species = boats_species[b]
            deploy_boats[-1][closest_task[b], t, species] += 1



#################
# Plotting code #
#################

if show_plots:
    #######################
    # Plot boat animation #
    #######################

    fig = plt.figure()

    ax = fig.add_subplot(111, autoscale_on=False,  aspect='equal', xlim=(0, arena_size), ylim=(0, arena_size))
    ax.grid()

    boats_line = []
    boats_current_line = []
    colors = ['b', 'g', 'r', 'c']
    for i in range(nspecies):
        boats_line.append(ax.plot([], [], colors[i] + '-', lw=1)[0])
        boats_current_line.append(ax.plot([], [], colors[i] + 'o', lw=1)[0])

    def plot_init():
        for i in range(nspecies):
            boats_line[i].set_data([], [])
            boats_current_line[i].set_data([], [])
        ax.plot(task_sites[:, 0], task_sites[:, 1], 'ro')
        for i in range(ntasks):
            circle = plt.Circle((task_sites[i, 0], task_sites[i, 1]), task_radius * 2., color='r', fill=False)
            ax.add_artist(circle)

    def plot_next_frame(i):
        ax.set_title('Time = %.2f[s] - %d' % (i * delta_t, i))
        for j in range(nspecies):
            boats_current_line[j].set_data(boats_pos[boats_species == j, i, 0], boats_pos[boats_species == j, i, 1])


    if show_movie:
        speedup = 32
        interval = int(1000. * delta_t / speedup)
        movie = animation.FuncAnimation(fig, plot_next_frame, np.arange(0, valid_timesteps),
                                    interval=interval, blit=False, repeat=False, init_func=plot_init)
        plt.show()
    
    else:
        plt.close(fig)
    if save_movie:
        movie.save(movie_filename, fps=1000. / interval)

    #########################################
    # Plot macroscopic model on top of boat #
    #########################################

    def compute_ratio_error(deploy_robots, deploy_traits_desired, transform, delta_t, match):
        num_tsteps = deploy_robots.shape[1]
        total_num_traits = np.sum(deploy_traits_desired)
        diffmac_rat = np.zeros(num_tsteps)
        for t in range(num_tsteps):
            if match==0:
                traits = np.dot(deploy_robots[:,t,:], transform)
                diffmac = np.abs(np.minimum(traits - deploy_traits_desired, 0))
            else:
                traits = np.dot(deploy_robots[:,t,:], transform)
                diffmac = (np.abs(traits - deploy_traits_desired)) / 2.0
            diffmac_rat[t] = np.sum(diffmac) / (total_num_traits)
        x = np.squeeze(np.arange(0, num_tsteps) * delta_t)
        return x, diffmac_rat

    def plot_traits_ratio_time(ax, x, diffmac_rat, color, label=None):
        ax.plot(x, diffmac_rat, color=color, linewidth=2, label=label)
        return fig

    fig = plt.figure(figsize = (6,3))
    ax = fig.add_subplot(111, autoscale_on=True)

    if auto_advance:
        for i in range(nslots):
            start_step = max(0, switching_steps[i])
            end_step = min(t_max, switching_steps[i + 1])
            duration = (end_step - start_step) * delta_t
            deploy_robots_euler = run_euler_integration(deploy_robots_init[i], Ks[i], duration, delta_t)
            x_mac, rat_mac = compute_ratio_error(deploy_robots_euler, deploy_traits_desired[i], species_traits, delta_t, match)
            plot_traits_ratio_time(ax, x_mac + start_step * delta_t, rat_mac, 'blue')
            rat_boat_runs = []
            for j in range(len(mat_run)):
                x_boat, rat_boat = compute_ratio_error(deploy_boats[j][:,start_step:end_step,:], deploy_traits_desired[i], species_traits, delta_t, match)
                rat_boat_runs.append(rat_boat.reshape(1, rat_boat.shape[0]))
            x_coords = x_boat + start_step * delta_t
            rat_boat_runs = np.concatenate(rat_boat_runs, axis=0)
            mean_rat = np.mean(rat_boat_runs, axis=0)
            plot_traits_ratio_time(ax, x_coords, mean_rat, 'green')
            # Add error bar.
            if len(mat_run) > 1:
                std_rat = np.std(rat_boat_runs, axis=0)
                if use_error_bars:
                    num_tsteps = len(x_coords)
                    err_ax = np.arange(0, num_tsteps, int(num_tsteps/5))
                    plt.errorbar(x_coords[err_ax],mean_rat[err_ax],std_rat[err_ax],fmt='o',markersize=3,color='green')
                else:
                    plt.fill_between(x_coords, mean_rat+std_rat, mean_rat-std_rat, facecolor='green', alpha=0.3)
            plt.legend(['Macroscopic', 'Boats'])
    else:
        deploy_robots_euler = run_euler_integration(deploy_robots_init[segment - 1], Ks[segment - 1], t_max, delta_t)
        x_mac, rat_mac = compute_ratio_error(deploy_robots_euler, deploy_traits_desired[segment - 1], species_traits, delta_t, match)
        x_boat, rat_boat = compute_ratio_error(deploy_boats[0], deploy_traits_desired[segment - 1], species_traits, delta_t, match)
        plot_traits_ratio_time(ax, x_mac, rat_mac, 'blue', 'Macroscopic')
        plot_traits_ratio_time(ax, x_boat, rat_boat, 'green', 'Boats')
        plt.legend()

    plt.xlim([0,t_max])
    plt.ylim([0,1])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Ratio of misplaced traits')

    plt.show()

    if save_data:
        prefix = 'processed_run_' + str(run) + '_'

        #pickle.dump(delta_t, open("delta_t", "wb"))
        #pickle.dump(t_max, open("t_max.p", "wb"))
        #pickle.dump(match, open("match.p", "wb"))
        pickle.dump(K, open(prefix+"K.p", "wb"))
        pickle.dump(deploy_robots_init, open(prefix+"deploy_robots_init.p", "wb"))
        pickle.dump(deploy_boats, open(prefix+"deploy_boats.p", "wb"))
        pickle.dump(deploy_robots_euler, open(prefix+"deploy_robots_euler.p", "wb"))
        pickle.dump(deploy_traits_desired, open(prefix+"deploy_traits_desired.p", "wb"))
        pickle.dump(species_traits, open(prefix+"species_traits.p", "wb"))


    if save_fig:
        fig.savefig('results_boats_evolution.eps') 