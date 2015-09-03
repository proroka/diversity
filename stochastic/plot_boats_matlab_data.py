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
from optimize_transition_matrix_hetero import *
from funcdef_macro_heterogeneous import *
from funcdef_micro_heterogeneous import *
from funcdef_util_heterogeneous import *
import funcdef_draw_network as nxmod

# Extra control vars.
save_movie = False
movie_filename = 'run.mp4'
show_movie = False
show_plots = True
remove_setup_time = True
remove_last_part = True

# Set the matlab workspace to load.
matlab_workspace_file = 'data/run_18_all_data.mat'

# Set these variables as in optimize_for_boats.py
match = 1
l_norm = 2

# Set the initial robot distribution used.
deploy_robots_init = np.array([[4, 0],
                               [0, 1],
                               [0, 0],
                               [0, 0]])

# Set the final robot distribution used.
deploy_robots_final = np.array([[0, 0],
                                [0, 0],
                                [0, 1],
                                [4, 0]])

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

if show_plots:
    #######################
    # Plot boat animation #
    #######################

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False,  aspect='equal', xlim=(0, arena_size), ylim=(0, arena_size))
    ax.grid()

    boats_line = []
    boats_current_line = []
    colors = ['b', 'g']
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
