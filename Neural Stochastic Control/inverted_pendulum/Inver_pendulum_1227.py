# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
# matplotlib.rcParams['text.usetex'] = True

from invert_pendulum_no_control_1227 import plot_fig1 as plot_fig1_no_control
from invert_pendulum_no_control_1227 import plot_fig2 as plot_fig2_no_control
from invert_pendulum_control_1227 import plot_fig1 as plot_fig1_control
from invert_pendulum_control_1227 import plot_fig2 as plot_fig2_control

def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)
    pass
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ax1 = plt.subplot(222)
    plot_fig1_no_control(ax1=ax1)
    plot_grid()
    ax2 = plt.subplot(231)
    plot_fig2_no_control(ax2=ax2)
    plot_grid()

    ax1 = plt.subplot(224)
    plot_fig1_control(ax1=ax1)
    plot_grid()

    ax2 = plt.subplot(234)
    plot_fig2_control(ax2=ax2)
    plot_grid()

    plt.show()