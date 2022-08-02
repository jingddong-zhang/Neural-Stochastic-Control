import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import math
import timeit
from scipy.integrate import odeint
from functions import *


def f(x,u=0):
    a, b, c = 1, 1, 1
    U2 = np.array([0.5, 0.74645887, 1.05370735, 0.38154169, 1.68833014, 0.83746371])
    x1, x2, x3, x4, x5, x6 = x+U2
    dx1 = 0.5 - a * x1
    dx2= 5 * x1 / ((1 + x1) * (1 + x3**4)) - b * x2
    dx3= 5 * x4 / ((1 + x4) * (1 + x2**4)) - c * x3
    dx4 = 0.5 / (1 + x2**4) - a * x4
    dx5 = (x1 * x4 / (1 + x1 * x4) + 4 * x3 / (1 + x3)) / (1 + x2**4) - a * x5
    dx6 = (x1 * x4 / (1 + x1 * x4) + 4 * x2 / (1 + x2)) / (1 + x3**4) - a * x6
    return np.array([dx1,dx2,dx3,dx4,dx5,dx6])

models = ControlNet(1,6,1)
models.load_state_dict(torch.load('./data/node_S_2.pkl'))

# models = ControlNet(2,12,2)
# models.load_state_dict(torch.load('./data/node_S.pkl'))

def run_0(n,dt,case,seed):
    np.random.seed(seed)
    U2 = np.array([0.5, 0.74645887, 1.05370735, 0.38154169, 1.68833014, 0.83746371])
    x0 = np.array([0.5,-0.9,0.6,-0.6,-0.9,0.5])
    X = np.zeros([n,6])
    DU = np.zeros([n-1,6])
    SU = np.zeros([n-1,6])
    X[0,:]=x0
    z = np.random.normal(0,1,n) # common noise
    # z = np.random.normal(0, 1, [n,6])  # common noise
    for i in range(n-1):
        x = X[i,:]
        df = f(x)
        if case == 0:
            X[i+1,:] = x+df*dt
        if case == 'S':
            with torch.no_grad():
                input = torch.from_numpy(x).to(torch.float32).unsqueeze(0)
                u = models(input).detach().numpy()
            X[i+1,:]=x+df*dt
            # X[i+1:i+2,1:3] += np.sqrt(dt)*z[i]*(u)
            X[i + 1:i + 2, 1:2] += np.sqrt(dt) * z[i] * (u)

    return X

'''
data generate
'''

seed = 3
n = 50000
# dt = 0.00001
dt = 0.0003
m = 10
# X = np.zeros([11,5000,6])
# X[0,:] = run_0(5000,0.001,0,0)
# for i in range(10):
#     X[i+1,:] = run_0(50000,dt,'S',i)[0:50000:10,:]
#     print(i)
# np.save('./data/pin_control_2',X)


'''
test
'''
# X = run_0(n,dt,'S',1)
# for i in range(6):
#     plt.plot(np.arange(len(X))*dt,X[:,i],label=r'$x_{}$'.format(i))
# plt.legend()



'''
plot
'''
font_size = 20


def subplot(X,xticks1,xticks2,yticks1,yticks2,ylim,title):
    alpha = 0.5
    mean_x,std_x,mean_y,std_y=np.mean(X[:,:,0],axis=0),np.std(X[:,:,0],axis=0),np.mean(X[:,:,1],axis=0),np.std(X[:,:,1],axis=0)
    length = len(mean_x)
    plt.fill_between(np.arange(length),mean_x-std_x,mean_x+std_x,color=colors[0],alpha=alpha)
    plt.plot(np.arange(length),mean_x,color=colors[0],label=r'$x$')
    plt.fill_between(np.arange(length),mean_y-std_y,mean_y+std_y,color=colors[1],alpha=alpha)
    plt.plot(np.arange(length),mean_y,color=colors[1],label=r'$y$')
    plot_grid()
    plt.legend(fontsize=font_size)
    plt.xticks(xticks1,xticks2,fontsize=font_size)
    plt.yticks(yticks1,yticks2,fontsize=font_size)
    plt.ylim(ylim)
    plt.title('{}'.format(title),fontsize=font_size)
    plt.xlabel('Time',fontsize=font_size)



def plot(alpha=0.5):
    data = np.load('./data/pin_control_2.npy')
    plt.subplot(121)
    X=data[0,:]
    for i in range(6):
        plt.plot(np.arange(len(X))*0.001,X[:,i],color=colors[i],label=r'$x_{}$'.format(i))
    # plt.legend(fontsize=font_size*0.7,ncol=3)
    plt.ylabel('State variables',fontsize=font_size)
    plt.xlabel('Time', fontsize=font_size)
    plt.yticks([-2, 0, 2],fontsize=font_size)
    plt.xticks([0, 2.5, 5.0], fontsize=font_size)
    plot_grid()
    # plt.legend(fontsize=font_size*0.7 , ncol=6, bbox_to_anchor=(1.5, 1.1))
    plt.subplot(122)
    X=data[1:,:]
    for i in range(6):
        x = X[:,:,i]
        mean_x = np.mean(x,axis=0)
        std_x = np.mean(x,axis=0)
        length = len(mean_x)
        plt.fill_between(np.arange(length)*0.003, mean_x - std_x, mean_x + std_x, color=colors[i], alpha=alpha)
        plt.plot(np.arange(length)*0.003, mean_x, color=colors[i], label=r'$x_{}$'.format(i))

    plt.xticks([0,15],fontsize=font_size)
    plt.yticks([-2,0,2],fontsize=font_size)
    plt.ylim(-2,2)
    # plt.ylabel('state variables',fontsize=font_size)
    plt.xlabel('Time', fontsize=font_size)
    plot_grid()


plot()
plt.show()