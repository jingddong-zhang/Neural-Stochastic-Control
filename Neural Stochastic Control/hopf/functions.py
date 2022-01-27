import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec


#向量场
def f(y,t) :
    #parameters  
    x1,x2 = y    
    dydt = [-25.0*x1-x2+x1*(x1**2+x2**2),x1-25*x2+x2*(x1**2+x2**2)] 
    return dydt

#绘制向量场
def Plotflow(Xd, Yd, t):
    # Plot phase plane 
    DX, DY = f([Xd, Yd],t)
    DX=DX/np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
    DY=DY/np.linalg.norm(DY, ord=2, axis=1, keepdims=True)
    plt.streamplot(Xd,Yd,DX,DY, color=('gray'), linewidth=0.5,
                  density=0.6, arrowstyle='-|>', arrowsize=1.5)



def plot_orbit(ax,title,path='./hopf/control_data.pt'):
    data = torch.load(path)
    X = data['X'].clone().detach()
    X1 = data['X1'].clone().detach()
    X2 = data['X2'].clone().detach()

    #添加极限环
    C = plt.Circle((0, 0),5, color='g', linewidth=2.5, fill=False)
    ax.add_artist(C)

    #绘制向量场
    xd = np.linspace(-10, 10, 10) 
    yd = np.linspace(-10, 10, 10)
    Xd, Yd = np.meshgrid(xd,yd)
    t = np.linspace(0,2,2000)
    Plotflow(Xd, Yd,t) 

    m = len(X1)
    for i in range(m):
        if 9.6 > X[i,0] > 5.5 and torch.max(X[i,:])<10 and torch.min(X[i,:])>0: #避免扰动过大的轨道出现
            plt.plot(X1[i,0],X2[i,0],marker='*',markersize=8,color='r')
            plt.plot(X1[i,:],X2[i,:],linestyle='--',color='r')
        elif X[i,0] < 4.5 and torch.max(X[i,:])<10 and torch.min(X[i,:])>0:   #避免扰动过大的轨道出现
            plt.plot(X1[i,0],X2[i,0],marker='*',markersize=8,color='b')
            plt.plot(X1[i,:],X2[i,:],linestyle='--',color='b')

    plt.legend([C],['limit cycle'],loc='upper right')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

#绘制极限环外部出发的轨道
def uncontrol_trajectory1(ax,title,path='./hopf/data.pt'):
    data = torch.load(path)
    X = data['X']
    C = plt.axhline(y=5.0,ls="--",linewidth=2.5,color="green")#添加水平直线
    U = plt.axhline(y=9.5,ls="--",linewidth=2.5,color="black")
    ax.add_artist(C)
    ax.add_artist(U)
    for i in range(len(X)):
        if 9.5 > X[i,0] > 5.5:
            x = X[i,:].numpy()
            m = np.max(x)
            index = np.argwhere(x == m )
            sample_length = int(index[0])
            L = np.arange(len(X[0,:sample_length]))
            plt.plot(L[0],X[i,0],marker='*',markersize=8,color='r')
            plt.plot(L,X[i,:sample_length],linestyle='--',color='r')
    plt.legend([U,C],[r'$\rho$=9.5',r'$\rho$=5.0'],borderpad=0.01, labelspacing=0.01)
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel(r'$\rho$')

#绘制极限环内部出发的轨道，sample_length的作用是从data中选择适当的轨道长度绘图
def uncontrol_trajectory2(ax,title,sample_length = 40,path='./hopf/control_data.pt'):
    data = torch.load(path)
    X = data['X'].clone().detach()
    C = plt.axhline(y=5.0,ls="--",linewidth=2.5,color="green")      #添加水平直线，对应极限环
    U = plt.axhline(y=0.0,ls="--",linewidth=2.5,color="deeppink")   #添加水平直线，对应零点
    ax.add_artist(C)
    ax.add_artist(U)
    for i in range(len(X)):
        if X[i,0] < 4.5:
            L = np.arange(len(X[0,:sample_length]))
            plt.plot(L[0],X[i,0],marker='*',markersize=8,color='b')
            plt.plot(L,X[i,:sample_length],linestyle='--',color='b')
    plt.legend([C,U],[r'$\rho$=5.0',r'$\rho$=0.0'],borderpad=0.01, labelspacing=0.01)
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel(r'$\rho$')


#绘制控制下的极限环外部出发的轨道
def control_trajectory1(ax,title,sample_length,path='./hopf/data.pt'):
    data = torch.load(path)
    X = data['X'].clone().detach()
    C = plt.axhline(y=5.0,ls="--",linewidth=2.5,color="green")#添加水平直线
    ax.add_artist(C)
    for i in range(len(X)):
        if 9.6 > X[i,0] > 5.5:
            L = np.arange(len(X[0,:sample_length]))
            plt.plot(L[0],X[i,0],marker='*',markersize=8,color='r')
            plt.plot(L,X[i,:sample_length],linestyle='--',color='r')
    plt.legend([C],[r'$\rho$=5.0'],borderpad=0.01, labelspacing=0.01)
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel(r'$\rho$')

#绘制控制下的极限环内部出发的轨道
def control_trajectory2(ax,title,sample_length = 40,path='./hopf/control_data.pt'):
    data = torch.load(path)
    X = data['X'].clone().detach()
    C = plt.axhline(y=5.0,ls="--",linewidth=2.5,color="green")#添加水平直线
    ax.add_artist(C)
    for i in range(len(X)):
        if X[i,0] < 4.5:
            L = np.arange(len(X[0,:sample_length]))
            plt.plot(L[0],X[i,0],marker='*',markersize=8,color='b')
            plt.plot(L,X[i,:sample_length],linestyle='--',color='b')
    plt.legend([C],[r'$\rho$=5.0'],borderpad=0.01, labelspacing=0.01)
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel(r'$\rho$')



