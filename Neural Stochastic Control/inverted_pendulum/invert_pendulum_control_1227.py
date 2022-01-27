import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from functions import *

from base_function import colors

alpha = 1.0
fontsize=35
fontsize_legend = 20
MarkerSize = 60
linewidth = 5
color_w = 0.15 #0.5
framealpha = 0.7
N_seg = 100


def plt_tick_1():
    # plt.ylim([-2.5, 2.5])
    # plt.xlim([-2.5, 2.5])
    # plt.xticks([-5, -2.5, 0, 2.5, 5], ['$-5$', '', '$0$', '', '$5$'])
    # plt.yticks([-5, -2.5, 0, 2.5, 5], ['$-5$', '', '$0$', '', '$5$'])
    plt.xticks([-10, -5, 0, 5, 10], ['$-10$', '', '$0$', '', '$10$'])
    plt.yticks([-10, -5, 0, 5, 10], ['$-10$', '', '$0$', '', '$10$'])
def plt_tick_2():
    # plt.ylim([-2.5, 2.5])
    plt.xticks([0, 0.075, 0.15, 0.225, 0.3], ['$0$', '', '$0.15$', '', '$0.3$'])
    plt.yticks([-10, -5, 0, 5, 10], ['$-10$', '', '$0$', '', '$10$'])


def plot_jianbian_line(
        X, Y, start_color=np.array([1.0, 0.0, 0.0]),
        end_color=np.array([0.0, 1.0, 0.0]),
        scale = 1/3,
        width_rate = 9/10,
):
    # start_color = 1 - start_color
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
        plt.plot(X[max(seg_len * i - 1, 0):seg_len * (i+1)], Y[max(seg_len * i - 1, 0):seg_len * (i+1)],
                 '-', color=now_color, alpha=alpha, linewidth= linewidth - w * linewidth * width_rate )



#五次倒立摆实验，angle和velocity分别保存为X1，X2
data = torch.load('./control_data.pt')

X1 = data['X1'].clone().detach() #data size=[5,10000]
X2 = data['X2'].clone().detach() #data size=[5,10000]


# fig = plt.figure()
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

# ax1 = plt.subplot(121)
show_indx = [0, 2, 4]


def plot_fig1(ax1):
    xd = np.linspace(-10, 10, 20)
    yd = np.linspace(-10, 10, 20)
    Xd, Yd = np.meshgrid(xd,yd)
    Plotflow(Xd, Yd)      #绘制向量场

    # #添加水平直线
    # C1 = plt.scatter(0,0,marker='o',color='g')
    # C2 = plt.scatter(math.pi,0,marker='o',color='r')
    # C3 = plt.scatter(-math.pi,0,marker='o',color='b')
    # ax1.add_artist(C1)
    # ax1.add_artist(C2)
    # ax1.add_artist(C3)


    color_id = 0
    # for i in range(2):
    for i in show_indx:
        # plt.plot(X1[i,0],X2[i,0],marker='*',color=cm.Accent(i*2))
        # plt.plot(X1[i,:2000],X2[i,:2000],color=cm.Accent(i*2),alpha=0.95)   #选择合适的长度

        plot_jianbian_line(X=X1[i,:2000], Y=X2[i,:2000], start_color=colors[color_id] * color_w, end_color=colors[color_id],
                           scale=1/3, width_rate=0.5)
        # plt.plot(state[0,0],state[1,0],marker='*', color=cm.Accent(i*2))
        color_id += 1

    color_id = 0
    for i in show_indx:
        # plt.scatter(X1[i,0],X2[i,0], marker='*', s=MarkerSize * 5, color='k', zorder=10)
        # plt.scatter(X1[i,0],X2[i,0], marker='*', s=MarkerSize * 5, color=colors[color_id] * color_w, zorder=10)
        plt.scatter(X1[i,0],X2[i,0], marker='*', s=MarkerSize * 5, color=colors[color_id]/max(colors[color_id]) * 0.7, zorder=10)
        color_id += 1

    #添加水平轴
    C1 = plt.scatter(0, 0,marker='o',color='g', s=MarkerSize, zorder=10)
    C2 = plt.scatter(math.pi,0,marker='o',color='r', s=MarkerSize, zorder=10)
    C3 = plt.scatter(-math.pi,0,marker='o',color='b', s=MarkerSize, zorder=10)
    ax1.add_artist(C1)
    ax1.add_artist(C2)
    ax1.add_artist(C3)
    plt.xlim(-6,6)
    plt.ylim(-6,6)

    # plt.title('Orbits under Stochastic Control')
    plt.legend([C1,C2,C3],[r'$(0,~0)$',r'$(\pi,~0)$',r'$(-\pi,~0)$'],loc='upper right',
               borderpad=0.05, labelspacing=0.05,fontsize=fontsize_legend, framealpha=framealpha)
    plt.xlabel(r'$\theta$',fontsize=fontsize)
    plt.ylabel(r'$\dot{\theta}$',fontsize=fontsize)
    plt_tick_1()
    plt.tick_params(labelsize=fontsize)




N_data = 3000
def control_trajectory_(ax,title,path='./control_data.pt'):
    data = torch.load(path)
    # X = data['X'].clone().detach()
    X1 = data['X1'].clone().detach()
    print('X1 shape:{}'.format(X1.shape))
    # X2 = data['X2']

    L1 = plt.axhline(y=0.0,ls="--",linewidth=1.5,color="green")#添加水平直线
    L2 = plt.axhline(y=math.pi,ls="--",linewidth=1.5,color="r")
    L3 = plt.axhline(y=-math.pi,ls="--",linewidth=1.5,color="b")
    ax.add_artist(L1)
    ax.add_artist(L2)
    ax.add_artist(L3)

    color_id = 0
    # for i in range(len(X1)):
    for i in show_indx:
        # x = X[i,:].numpy()
        # m = np.max(x)
        # index = np.argwhere(x == m )
        # sample_length = int(index[0])
        L = np.arange(len(X1[0,:N_data])) * 0.0001
        # plt.plot(L[0],X1[i,0],marker='*',markersize=8,color=cm.Accent(i*2))
        plot_jianbian_line(X=L, Y=X1[i, :N_data],
                           start_color=colors[color_id] * color_w, end_color=colors[color_id],
                           scale = 1/2,
                           width_rate = 5/10,
                           )
        # plt.plot(L,X1[i,:3000],linestyle='--',color=cm.Accent(i*2),alpha=0.45)
        color_id += 1
    color_id = 0
    for i in show_indx:
        # plt.scatter(L[0],X1[i,0],marker='*', s=MarkerSize * 5, color=colors[color_id] * color_w, zorder=10)
        plt.scatter(L[0],X1[i,0],marker='*', s=MarkerSize * 5, color=colors[color_id]/max(colors[color_id]) * 0.7, zorder=10)

        color_id += 1
    plt.legend([L1,L2,L3],[r'$\theta=0$',r'$\theta=\pi$',r'$\theta=-\pi$'],loc='upper right',
               borderpad=0.05, labelspacing=0.05, fontsize=fontsize_legend, framealpha=framealpha)
    # plt.title(title)
    plt.xlabel('Time',fontsize=fontsize)
    plt.ylabel(r'$\theta$',fontsize=fontsize)


# ax2 = plt.subplot(122)
def plot_fig2(ax2):
    # control_trajectory(ax2,'Phase Trajectories along Time','./control_data.pt')
    control_trajectory_(ax2,'Phase Trajectories along Time','./control_data.pt')


    plt_tick_2()
    plt.tick_params(labelsize=fontsize)





if __name__ == '__main__':
    ax1 = plt.subplot(121)
    plot_fig1(ax1=ax1)
    ax2 = plt.subplot(122)
    plot_fig2(ax2=ax2)
    plt.show()