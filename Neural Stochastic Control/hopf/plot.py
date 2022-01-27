import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functions import *


def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)


if __name__ == '__main__':

    max_len = 6
    fig = plt.figure()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    gs = gridspec.GridSpec(16, 13)
    ax1 = plt.subplot(gs[0:7, 9:13])
    plot_orbit(ax1,'Phase Orbits','./data.pt')
    plot_grid()

    ax2 = plt.subplot(gs[0:3,0:max_len])
    # plot_orbit(ax1,'Phase Orbits under Stochastic Control','./neural_sde/hopf/control_data.pt')
    uncontrol_trajectory1(ax2,'Plot along Trajectories',path='./data.pt')
    plot_grid()

    ax3 = plt.subplot(gs[4:7,0:max_len])
    uncontrol_trajectory2(ax3,None,200,path='./data.pt')
    plot_grid()

    ax4 = plt.subplot(gs[9:16, 9:13])
    plot_orbit(ax4,None,'./control_data.pt')
    plot_grid()

    ax5 = plt.subplot(gs[9:12,0:max_len])
    control_trajectory1(ax5,None,40,path='./control_data.pt')
    plot_grid()

    ax6 = plt.subplot(gs[13:16,0:max_len])
    control_trajectory2(ax6,None,40,path='./control_data.pt')
    # control_trajectory2(ax5,None,40,path='./control_data.pt')
    plot_grid()


    plt.show()