import numpy as np
import scipy as sp
import scipy.io
import scipy.ndimage.filters
import pylab as pl
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
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

# Extra control vars.
save_movie = False
save_all_movie_frames = False
movie_filename = 'run.mp4'
show_movie = False
show_trait_ratio = False
show_overall_plot = True
overall_plot_filename = 'overall.eps'
show_empty_arena = False
remove_setup_time = True
remove_last_part = True

# Vars for the overall plot.
overall_trail_size = 6.0 * 300  # In time-steps, 1 mins == 60 / 0.2 == 300
overall_remove_last_timestamps = 0.1 * 300  # Number of timestamps to remove from the end of the run.

# Set the matlab workspace to load.
matlab_workspace_file = 'data/boats/run_25_all_data.mat'

# Set these variables as in optimize_for_boats.py
match = 1
l_norm = 2

# Set the initial robot distribution used.
deploy_robots_init = np.array([[2, 0],
                               [0, 2],
                               [0, 0],
                               [0, 0]])

# Set the final robot distribution used.
deploy_robots_final = np.array([[0, 0],
                                [0, 0],
                                [0, 2],
                                [2, 0]])

# Set the species-trait matrix used.
species_traits = np.array([[1, 0],
                           [0, 1]])


####################
# Rest of the code #
####################

print 'Number of robots =', np.sum(deploy_robots_init)
print 'Number of robots per species =', np.sum(deploy_robots_init, axis=0)
assert np.sum(np.abs(np.sum(deploy_robots_init, axis=0) - np.sum(deploy_robots_final, axis=0))) == 0, 'Number of robots is different between initial and final distribution'

# Get trait distributions.
deploy_traits_init = np.dot(deploy_robots_init, species_traits)
deploy_traits_desired = np.dot(deploy_robots_final, species_traits)

# Load mat file and get relevant matrices.
matlab_data = sp.io.loadmat(matlab_workspace_file)

task_sites = matlab_data['task_sites']
ntasks = task_sites.shape[0]

boats_species = np.squeeze(matlab_data['boats_species']) - 1
nboats = boats_species.shape[0]
nspecies = matlab_data['nspecies']
ntraits = np.max(boats_species) + 1

# Quick sanity checks.
assert deploy_robots_init.shape[0] == ntasks, 'Wrong number of tasks'
assert deploy_robots_init.shape[1] == nspecies, 'Wrong number of species'
assert deploy_robots_final.shape[0] == ntasks, 'Wrong number of tasks'
assert deploy_robots_final.shape[1] == nspecies, 'Wrong number of species'
assert species_traits.shape[0] == nspecies, 'Wrong number of species'
assert species_traits.shape[1] == ntraits, 'Wrong number of traits'

t_max = matlab_data['max_time']
delta_t = matlab_data['dt']
setup_time = matlab_data['setup_time']

# Build K as in the python code.
K = np.empty((ntasks, ntasks, nspecies))
for s, k in enumerate(matlab_data['K']):
    K[:, :, s] = k[0]

arena_size = matlab_data['arena_size'][0][0]
task_radius = matlab_data['task_radius']
boats_pos = matlab_data['boats_pos']
print 'Number of time-steps =', boats_pos.shape[1]
# Remove setup-time.
if remove_setup_time:
    boats_pos = boats_pos[:, int(setup_time / delta_t):, :]
# Smooth boats positions.
valid_timesteps = boats_pos.shape[1]
# Remove end of run (if Ctrl-C was pressed for example).
if remove_last_part:
    for t in range(boats_pos.shape[1]):
        if np.all(boats_pos[:, t, :] == 0.):
            valid_timesteps = t
            break
    boats_pos = boats_pos[:, :valid_timesteps, :]
for i in range(nboats):
    boats_pos[i, :, 0] = sp.ndimage.filters.median_filter(boats_pos[i, :, 0], size=10, mode='reflect')
    boats_pos[i, :, 1] = sp.ndimage.filters.median_filter(boats_pos[i, :, 1], size=10, mode='reflect')


# Compute per timesteps the number of boats in each task.
deploy_boats = np.zeros((ntasks, valid_timesteps, nspecies))
ntimesteps = boats_pos.shape[1]
print 'Number of actual time-steps =', ntimesteps
for t in range(ntimesteps):
    positions = boats_pos[:, t, :]
    for b in range(nboats):
        position = positions[b, :]
        closest_task = np.argmin(np.sum(np.square(task_sites - position), axis=1))
        species = boats_species[b]
        deploy_boats[closest_task, t, species] += 1

#################
# Plotting code #
#################

if show_movie or save_movie:
    #######################
    # Plot boat animation #
    #######################

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False,  aspect='equal', xlim=(0, arena_size), ylim=(0, arena_size))
    ax.grid()

    trail_size = 50
    boats_line = []
    boats_current_line = []
    colors = ['b', 'g']
    colormaps = ['Blues', 'Greens']
    for i in range(nspecies):
        boats_current_line.append(ax.plot([], [], colors[i] + 'o', lw=1, markersize=8)[0])
    for i in range(nboats):
        lc = LineCollection(np.array([[[0, 0],[0, 0]]]), cmap=plt.get_cmap(colormaps[boats_species[i]]),
                            norm=plt.Normalize(0, trail_size), linewidths=np.linspace(1, 8, trail_size))
        boats_line.append(lc)
        lc.set_linewidth(3)
        ax.add_collection(lc)

    def plot_init():
        for i in range(nspecies):
            boats_current_line[i].set_data([], [])
        for i in range(nboats):
            boats_line[i].set_segments(np.array([[[0, 0],[0, 0]]]))
        for i in range(ntasks):
            circle = plt.Circle((task_sites[i, 0], task_sites[i, 1]), task_radius * 2., color='r', fill=False, lw=2)
            ax.add_artist(circle)

    def plot_next_frame(i):
        ax.set_title('Time = %.2f[s] - %d' % (i * delta_t, i))
        for j in range(nspecies):
            boats_current_line[j].set_data(boats_pos[boats_species == j, i, 0], boats_pos[boats_species == j, i, 1])
        for j in range(nboats):
            x = boats_pos[j, max(i-trail_size, 0):i, 0]
            y = boats_pos[j, max(i-trail_size, 0):i, 1]
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            boats_line[j].set_array(np.arange(trail_size))
            boats_line[j].set_segments(segments)
            boats_line[j].set_linewidths(np.linspace(1, 8, trail_size - 1))
            if save_all_movie_frames:
                plt.savefig('plot_outputs/img_%05d.png' % i, bbox_inches='tight')


    speedup = 32
    interval = int(1000. * delta_t / speedup)
    movie = animation.FuncAnimation(fig, plot_next_frame, np.arange(0, valid_timesteps),
                                    interval=interval, blit=False, repeat=False, init_func=plot_init)
    if save_movie:
        movie.save(movie_filename, fps=1000. / interval)
    if show_movie:
        plt.show()
    else:
        plt.close(fig)

if show_empty_arena:
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False,  aspect='equal', xlim=(0, arena_size), ylim=(0, arena_size))
    ax.grid()
    for i in range(ntasks):
        circle = plt.Circle((task_sites[i, 0], task_sites[i, 1]), task_radius * 2., color='g', fill=False, lw=2)
        ax.add_artist(circle)
    plt.show()

if show_trait_ratio:
    #########################################
    # Plot macroscopic model on top of boat #
    #########################################

    def plot_traits_ratio_time(ax, deploy_robots, deploy_traits_desired, transform, delta_t, match, color, label):
        num_tsteps = deploy_robots.shape[1]
        total_num_traits = np.sum(deploy_traits_desired)
        diffmac_rat = np.zeros(num_tsteps)
        for t in range(num_tsteps):
            if match==0:
                traits = np.dot(deploy_robots[:,t,:], transform)
                diffmac = np.abs(np.minimum(traits - deploy_traits_desired, 0))
            else:
                traits = np.dot(deploy_robots[:,t,:], transform)
                diffmac = np.abs(traits - deploy_traits_desired)
            diffmac_rat[t] = np.sum(diffmac) / total_num_traits
        x = np.squeeze(np.arange(0, num_tsteps) * delta_t)
        l2 = ax.plot(x, diffmac_rat, color=color, linewidth=2, label=label)
        return fig

    # Simulate macro.
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=True)
    deploy_robots_euler = run_euler_integration(deploy_robots_init, K, t_max, delta_t)
    plot_traits_ratio_time(ax, deploy_robots_euler, deploy_traits_desired, species_traits, delta_t, match, 'blue', 'Macroscopic')
    plot_traits_ratio_time(ax, deploy_boats, deploy_traits_desired, species_traits, delta_t, match, 'green', 'Boats')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Ratio of misplaced traits')
    plt.legend()
    plt.show()

if show_overall_plot:
    #########################################################
    # Show an overall plot of the trajectories.             #
    # A bit ugly - as it is copy-pasted from the movie code #
    #########################################################

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False,  aspect='equal', xlim=(0, arena_size), ylim=(0, arena_size))
    ax.grid()
    boats_line = []
    boats_current_line = []
    colors = ['r', 'g']
    colormaps = ['Reds', 'Greens']
    for i in range(ntasks):
        circle = plt.Circle((task_sites[i, 0], task_sites[i, 1]), task_radius * 2., color='black', fill=False, lw=2)
        ax.add_artist(circle)
    for i in range(nspecies):
        boats_current_line.append(ax.plot([], [], colors[i] + 'o', lw=1, markersize=8)[0])
    for i in range(nboats):
        lc = LineCollection(np.array([[[0, 0], [0, 0]]]), cmap=plt.get_cmap(colormaps[boats_species[i]]),
                            norm=plt.Normalize(0, overall_trail_size), linewidths=np.linspace(1, 8, overall_trail_size))
        boats_line.append(lc)
        lc.set_linewidth(3)
        ax.add_collection(lc)
    for j in range(nspecies):
        boats_current_line[j].set_data(boats_pos[boats_species == j, -overall_remove_last_timestamps, 0],
                                       boats_pos[boats_species == j, -overall_remove_last_timestamps, 1])
    for j in range(nboats):
        x = boats_pos[j, -overall_remove_last_timestamps-overall_trail_size:-overall_remove_last_timestamps, 0]
        y = boats_pos[j, -overall_remove_last_timestamps-overall_trail_size:-overall_remove_last_timestamps, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        color_offset = overall_trail_size / 5
        boats_line[j].set_array(np.arange(color_offset, overall_trail_size + color_offset))
        boats_line[j].set_segments(segments)
        boats_line[j].set_linewidths(np.linspace(2, 8, overall_trail_size - 1))
    if overall_plot_filename:
        plt.savefig(overall_plot_filename, bbox_inches='tight')
    plt.show()
