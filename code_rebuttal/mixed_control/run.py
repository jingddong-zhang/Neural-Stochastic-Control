import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import math
import timeit
from scipy.integrate import odeint
from functions import *
from cvxopt import solvers,matrix


def f(x,u=0):
    u,v = x
    G = 9.81  # gravity
    L = 0.5   # length of the pole
    m = 0.15  # ball mass
    b = 0.1   # friction
    return np.array([v,G*np.sin(u)/L +(-b*v)/(m*L**2)])



models = Net(2,6,2)
models.load_state_dict(torch.load('./data/S.pkl'))
modeld = Net(2,6,2)
modeld.load_state_dict(torch.load('./data/D.pkl'))
modelmd = Net(2,6,2)
modelmd.load_state_dict(torch.load('./data/MD.pkl'))
modelms = Net(2,6,2)
modelms.load_state_dict(torch.load('./data/MS.pkl'))

def run_0(n,dt,case,seed):
    np.random.seed(seed)
    x0 = np.array([3.0,-4.0])
    X = np.zeros([n,2])
    DU = np.zeros([n-1,2])
    SU = np.zeros([n-1,2])
    X[0,:]=x0
    z = np.random.normal(0,1,n) # common noise
    # z = np.random.normal(0,1,[n,4]) # uncorrelated noise

    for i in range(n-1):
        x = X[i,:]
        df = f(x)
        if case == 0:
            X[i+1,:] = x+df*dt#+()*(dt*z[i]**2-dt)/(2*np.sqrt(dt))
        if case == 'S':
            with torch.no_grad():
                input = torch.from_numpy(x).to(torch.float32).unsqueeze(0)
                u = models(input).detach().numpy()
            X[i+1,:]=x+df*dt+np.sqrt(dt)*z[i]*(u)
            SU[i,:] = u
        if case == 'D':
            with torch.no_grad():
                input = torch.from_numpy(x).to(torch.float32).unsqueeze(0)
                u = modeld(input).detach().numpy()
            X[i + 1, :] = x + (df+u) * dt
            DU[i, :] = u
        if case == 'M':
            with torch.no_grad():
                input = torch.from_numpy(x).to(torch.float32).unsqueeze(0)
                d_u = modelmd(input).detach().numpy()
                s_u = modelms(input).detach().numpy()
            X[i+1,:]=x+(df+d_u)*dt+np.sqrt(dt)*z[i]*(s_u)
            DU[i,:] = d_u
            SU[i,:] = s_u
    return X,DU,SU

'''
data generate
'''
seed = 3
n = 50000
dt = 0.00001
m = 10
# X,DU,SU = np.zeros([m,n,2]),np.zeros([m,n-1,2]),np.zeros([m,n-1,2])
# for i in range(m):
#     X[i,:],DU[i,:],SU[i,:] = run_0(n,dt,'D',2*i+1)
#     print(i)
# np.save('./data/S.npy',{'X':X,'DU':DU,'SU':SU}) # (5000,0.0001)
# np.save('./data/M.npy',{'X':X,'DU':DU,'SU':SU}) # throw out 2nd trajectory (5000,0.0001)
# np.save('./data/D.npy',{'X':X,'DU':DU,'SU':SU})


def energy(U,n=5000,dt=0.0001):
    n = n-1
    a=np.linspace(0,dt*(n-1),n)
    e = 0.0
    for i in range(len(U)):
        e += integrate.trapz(np.array(np.sum(U[i,:]**2,axis=1)),a)
    return e/float(len(U))

def stop_time(X,delta=0.001,dt=0.0001):
    time = 0
    for i in range(len(X)):
        norm_x = np.sqrt(X[i,:,0]**2+X[i,:,1]**2)
        index = np.where(norm_x<delta)
        time += index[0][0]
    return time/float(len(X))*dt

def minima(X):
    min_x = 0
    for i in range(len(X)):
        norm_x = np.sqrt(X[i,:,0]**2+X[i,:,1]**2)
        min_x += np.min(norm_x)
        print(i,np.min(norm_x))
    return min_x/float(len(X))


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




def plot():
    plt.subplot(131)
    data = np.load('./data/D.npy',allow_pickle=True).item()
    X,DU,SU = data['X'],data['DU'],data['SU']
    X = X[:, 0:n:10, :]
    subplot(X,[0,2000,4000],[0,0.2,0.4],[-2,0,2,4],[-2,0,2,4],[-2,5],'deterministic')
    plt.ylabel('state variables',fontsize=font_size)
    plt.title('ME:{}'.format(int(energy(DU+SU,n,dt))),fontsize=font_size)

    plt.subplot(132)
    data = np.load('./data/M.npy',allow_pickle=True).item()
    X,DU,SU = data['X'],data['DU'],data['SU']
    X = X[:,0:31000:10,:]
    subplot(X,[0,1500,3000],[0,0.15,0.3],[0,1,2],[0,'',2],[-0.2,2.5],'mix')
    plt.title('ME:{}'.format(int(energy(DU+SU,n,dt))),fontsize=font_size)

    plt.subplot(133)
    data = np.load('./data/S.npy',allow_pickle=True).item()
    X,DU,SU = data['X'],data['DU'],data['SU']
    X = X[:,0:31000:10,:]
    subplot(X,[0,1500,3000],[0,0.15,0.3],[0,1,2,3],[0,1,2,3],[-0.2,2.5],'stochastic')
    plt.title('ME:{}'.format(int(energy(DU+SU,n,dt))),fontsize=font_size)



plot()

plt.show()