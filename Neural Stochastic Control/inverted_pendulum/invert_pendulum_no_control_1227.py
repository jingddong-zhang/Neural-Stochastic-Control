import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.gridspec as gridspec
from functions import *
from base_function import colors
# colors = [
#     [233/256,	110/256, 236/256], # #e96eec
#     [223/256,	73/256,	54/256], # #df4936
#     [107/256,	161/256,255/256], # #6ba1ff
#     [0.6, 0.4, 0.8], # amethyst
#     [0.0, 0.0, 1.0], # ao
#     [0.55, 0.71, 0.0], # applegreen
#     # [0.4, 1.0, 0.0], # brightgreen
#     [0.99, 0.76, 0.8], # bubblegum
#     [0.93, 0.53, 0.18], # cadmiumorange
#     [0.6, 0.6, 0.2],  # olive
#     [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
#     [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
#     [11/255, 132/255, 147/255], # deblue
#     [204/255, 119/255, 34/255], # {ocra}
# ]




# colors = np.array(colors)


alpha = 1.0
fontsize=35
fontsize_legend = 20
MarkerSize = 60
linewidth = 5
color_w = 0.15 #0.5
framealpha = 0.7
N_seg = 100
def plt_tick_1():
    # plt.ylim([-10, 10])
    # plt.ylim([-2.5, 2.5])
    # plt.xlim([-2.5, 2.5])
    plt.xticks([-5, -2.5, 0, 2.5, 5], ['$-5$', '', '$0$', '', '$5$'])
    plt.yticks([-5, -2.5, 0, 2.5, 5], ['$-5$', '', '$0$', '', '$5$'])
def plt_tick_2():
    # plt.xticks([0, 2, 4, 6])
    plt.xticks([0, 1, 2, 3, 4], ['$0$', '', '$2$', '', '$4$'])
    plt.yticks([-5, -2.5, 0, 2.5, 5], ['$-5$', '', '$0$', '', '$5$'])



def plot_jianbian_line(
        X, Y, start_color=np.array([1.0, 0.0, 0.0]),
        end_color=np.array([0.0, 1.0, 0.0]),
        scale = 1/3,
        width_rate = 9/10,
):
    # start_color = 1- start_color
    start_color= end_color
    data_len = len(X)
    # plt.plot(data[0,:1000], data[1, :1000], '-', alpha=alpha)
    n = N_seg
    seg_len = data_len // n
    print('data_len:{}, n:{}, seg_len:{}'.format(data_len, n, seg_len))
    for i in range(n - 1):
        w = ((i) / n) ** (scale)
        now_color = start_color +  w * (end_color - start_color)
        # print('i:{}, now_color:{}'.format(i, now_color))
        # plt.plot(data[0,i:i+3], data[1,i:i+3], '-', color=now_color, alpha=alpha)
        plt.plot(X[seg_len * i:seg_len * (i+1)], Y[seg_len * i:seg_len * (i+1)],
                 '-', color=now_color, alpha=alpha, linewidth= linewidth - w * linewidth * width_rate )


np.random.seed(10)
t = np.arange(0.0, 4.0, 0.0001)
set_state0 = np.array([[-5.0,5.0],[-3.0,4.0],[-1.0,3.0],[1.0,-3.0],[3.0,-4.0],[5.0,-5.0]])
# fig = plt.figure()
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
# ax1 = plt.subplot(121)

# show_indx = [0, 2, 3, 5]
show_indx = [0, 2, 4]
def plot_fig1(ax1):

    xd = np.linspace(-5.5, 5.5, 10)
    yd = np.linspace(-5.5, 5.5, 10)
    Xd, Yd = np.meshgrid(xd,yd)
    Plotflow(Xd, Yd) #绘制向量场

    #添加水平轴


    # for i in range(6):
    color_id = 0
    for i in show_indx:
        # state0 = np.random.uniform(-6,6,2)
        state0 = set_state0[i,:]
        state = invert_pendulum(state0,t) #生成倒立摆轨迹
        # plt.plot(state[0,:],state[1,:],color=cm.Accent(i*2),alpha=0.55)

        plot_jianbian_line(X=state[0, :], Y=state[1, :], start_color=colors[color_id] * color_w, end_color=colors[color_id])
        # plt.plot(state[0,0],state[1,0],marker='*', color=cm.Accent(i*2))
        # plt.scatter(state[0,0],state[1,0], marker='*', s=MarkerSize * 5, color=1 - colors[color_id] * color_w)
        color_id += 1

    color_id = 0
    for i in show_indx:
        state0 = set_state0[i,:]
        state = invert_pendulum(state0,t) #生成倒立摆轨迹
        # plt.scatter(X1[i,0],X2[i,0], marker='*', s=MarkerSize * 5, color='k', zorder=10)
        # plt.scatter(X1[i,0],X2[i,0], marker='*', s=MarkerSize * 5, color=colors[color_id] * color_w, zorder=10)
        # plt.scatter(state[0,0],state[1,0], marker='*', s=MarkerSize * 5, color=colors[color_id] * color_w, zorder=10)
        plt.scatter(state[0,0],state[1,0], marker='*', s=MarkerSize * 5, color=colors[color_id]/max(colors[color_id]) * 0.7, zorder=10)

        color_id += 1

    #添加水平轴
    C1 = plt.scatter(0, 0,marker='o',color='g', s=MarkerSize, zorder=10)
    C2 = plt.scatter(math.pi,0,marker='o',color='r', s=MarkerSize, zorder=10)
    C3 = plt.scatter(-math.pi,0,marker='o',color='b', s=MarkerSize, zorder=10)
    ax1.add_artist(C1)
    ax1.add_artist(C2)
    ax1.add_artist(C3)

    # plt.title('Orbits along Vector Fields')
    plt.legend([C1,C2,C3],[r'$(0,~0)$', r'$(\pi,~0)$',r'$(-\pi,~0)$'],loc='upper right',borderpad=0.05, labelspacing=0.05,
               fontsize=fontsize_legend, framealpha=framealpha)
    # plt.xlabel(r'$\theta$', fontsize=fontsize)
    plt.ylabel(r'$\dot{\theta}$', fontsize=fontsize)
    plt_tick_1()
    plt.tick_params(labelsize=fontsize)

# ax2 = plt.subplot(122)
def plot_fig2(ax2):

    #添加水平轴
    L1 = plt.axhline(y=0.0,ls="--",linewidth=1.5,color="green")
    L2 = plt.axhline(y=math.pi,ls="--",linewidth=1.5,color="r")
    L3 = plt.axhline(y=-math.pi,ls="--",linewidth=1.5,color="b")
    ax2.add_artist(L1)
    ax2.add_artist(L2)
    ax2.add_artist(L3)


    color_id = 0
    # for i in range(6):
    for i in show_indx:
        # state0 = np.random.uniform(-6,6,2)
        state0 = set_state0[i,:]
        state = invert_pendulum(state0,t)  #生成倒立摆轨迹
        # plt.plot(t, state[0,:],color=cm.Accent(i**2+1),alpha=0.55)
        plot_jianbian_line(X=t, Y=state[0, :],
                           start_color=colors[color_id] * color_w, end_color=colors[color_id],
                           scale = 1/2,
                           width_rate = 5/10,
                           )
        # plt.plot(t[0],state[0,0],marker='*',color=cm.Accent(i**2+1))
        # plt.scatter(t[0],state[0,0],marker='*', s=MarkerSize * 5, color=1 - colors[color_id] * color_w)
        color_id += 1
    color_id = 0
    for i in show_indx:
        state0 = set_state0[i,:]
        state = invert_pendulum(state0,t) #生成倒立摆轨迹
        # plt.scatter(X1[i,0],X2[i,0], marker='*', s=MarkerSize * 5, color='k', zorder=10)
        # plt.scatter(X1[i,0],X2[i,0], marker='*', s=MarkerSize * 5, color=colors[color_id] * color_w, zorder=10)
        # plt.scatter(state[0,0],state[1,0], marker='*', s=MarkerSize * 5, color=colors[color_id] * color_w, zorder=10)
        # plt.scatter(t[0],state[0,0],marker='*', s=MarkerSize * 5, color=colors[color_id] * color_w, zorder=10)
        plt.scatter(t[0],state[0,0],marker='*', s=MarkerSize * 5, color=colors[color_id]/max(colors[color_id]) * 0.7, zorder=10)
        color_id += 1

    plt.legend(
        [L1,L2,L3],
        [r'$\theta=0$',r'$\theta=\pi$',r'$\theta=-\pi$'],loc='upper right',
        borderpad=0.05, labelspacing=0.05, fontsize=fontsize_legend, framealpha=framealpha
    )
    # plt.title('Phase Trajectories along Time')
    # plt.xlabel('t', fontsize=fontsize)
    plt.ylabel(r'$\theta$', fontsize=fontsize)
    plt_tick_2()
    plt.tick_params(labelsize=fontsize)


if __name__ == '__main__':
    ax1 = plt.subplot(121)
    plot_fig1(ax1=ax1)
    ax2 = plt.subplot(122)
    plot_fig2(ax2=ax2)
    plt.show()