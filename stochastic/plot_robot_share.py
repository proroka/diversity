from matplotlib import pyplot as plt
import matplotlib.colors
import matplotlib.cm
import numpy as np
import pickle


def colors_from(cmap_name, ncolors):
    cm = plt.get_cmap(cmap_name)
    cm_norm = matplotlib.colors.Normalize(vmin=0, vmax=ncolors - 1)
    scalar_map = matplotlib.cm.ScalarMappable(norm=cm_norm, cmap=cm)
    return [scalar_map.to_rgba(i) for i in range(ncolors)]


def plot_trait_share(deploy_robots, transform, delta_t=1., trait_index=0, cmap_name='jet', ylabel='trait'):
    t = np.arange(deploy_robots.shape[1]) * delta_t
    traits = np.empty((deploy_robots.shape[0], deploy_robots.shape[1], transform.shape[1]))
    for i in range(deploy_robots.shape[1]):
        traits[:, i, :] = deploy_robots[:, i, :].dot(transform)
    x = traits[:, :, trait_index]
    
    # square
    #fig, ax = plt.subplots(figsize=(4,4))
    # rectangular    
    fig, ax = plt.subplots(figsize=(8,4))    
    
    ax.stackplot(t, x, colors=colors_from(cmap_name, deploy_robots.shape[0]))
    ax.set_xlim([0, t[-1]])
    ax.set_ylim([0, np.sum(x[:, 0])])
    ax.yaxis.set_ticks([0, np.sum(x[:, 0])])
    ax.yaxis.set_ticklabels(['0', '1'])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Fraction of %s %d per task' % (ylabel, trait_index))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.patch.set_visible(False)
    ax.yaxis.tick_left()
    
    #plt.plot([7,7],[0,1], linewidth=2)
    
    return fig


def plot_robot_share(deploy_robots, delta_t=1., robot_index=0, cmap_name='jet'):
    return plot_trait_share(deploy_robots, np.identity(deploy_robots.shape[2]), delta_t=delta_t,
                            trait_index=robot_index, cmap_name=cmap_name, ylabel='species')


if __name__ == '__main__':
    # Shape is M x iterations x S
    deploy_robots = pickle.load(open('test_data.p'))
    for node in range(deploy_robots.shape[2]):
        plot_robot_share(deploy_robots, delta_t=0.04, robot_index=node, cmap_name='Spectral')
        plt.show()
    for node in range(2):
        plot_trait_share(deploy_robots, transform=np.ones((deploy_robots.shape[2], 2)), delta_t=0.04, trait_index=node, cmap_name='terrain')
        plt.show()
