import numpy as np
import matplotlib.pyplot as plt
from u_plot import *
from plot_trajectory import *
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
# matplotlib.rcParams['text.usetex'] = True
font_size = 15

'''
Pick trajectories data for corresponding $\alpha$ 
'''
A = torch.load('./data/hyper_a/data.pt')
A = A[:,-8:-1,:,:]
print(A.shape)



def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)


def plot_a(a):
    L = np.load('./data/hyper_a/a_{}.npy'.format(a))
    r_L =  np.zeros(1000-len(L))
    L = np.concatenate((L,r_L),axis=0)
    #  np.concatenate((a,b),axis=0)
    plt.plot(np.arange(len(L)),L,'b')
    # plt.xlabel('Iterations')
    plt.ylim(-0.01,1)
    plt.yticks([])
    plt.title(r'$\alpha={}$'.format(a))


for i in range(7):
    # plt.axes([0.1+0.17*i, 0.7, 0.1, 0.1])
    plt.subplot(4, 7, i+1)
    plot_a(float(format(0.65+i*0.05,'.2f')))
    plot_grid()
    if i == 0:
        plt.yticks([0,10,20])
        plt.ylabel('Loss',fontsize=font_size)
        plt.text(-5,5,'Training',rotation=90,fontsize=font_size)
    else:
        plt.yticks([0, 10, 20], ['', '', ''])
    if i == 3:
        plt.xlabel('Iterations',fontsize=font_size) 


for i in range(7):
    plt.subplot(4, 7, 7 + i+1)
    plot_trajec(A[0,i,:,0:60000:10],float(format(0.65+i*0.05,'.2f')))
    plot_grid()
    if i == 0:
        plt.yticks([-10,-5,0,5,10])
        plt.ylabel(r'$\theta$',fontsize=font_size)
        plt.text(-1,-5,'Trajectory',rotation=90,fontsize=font_size)
    else:
        plt.yticks([-10,-5, 0,5, 10], ['', '', '','',''])
    if i == 3:
        plt.xlabel('Time',fontsize=font_size) 

for i in range(7):
    plt.subplot(4, 7, 14 + i+1)
    plot_trajec(A[1,i,:,0:60000:10],float(format(0.65+i*0.05,'.2f')))
    plot_grid()
    if i == 0:
        plt.yticks([-10,-5,0,5,10])
        plt.ylabel(r'$\dot{\theta}$',fontsize=font_size)
        plt.text(-1,-5,'Trajectory',rotation=90,fontsize=font_size)
    else:
        plt.yticks([-10,-5, 0,5, 10], ['', '', '','',''])
    if i == 3:
        plt.xlabel('Time',fontsize=font_size) 


for i in range(7):
    # plt.axes([0.1+0.17*i, 0.1, 0.1, 0.1])
    plt.subplot(4, 7, 21 + i+1)
    draw(float(format(0.65+i*0.05,'.2f')))
    if i == 0:
        plt.yticks([-5,0,5])
        plt.ylabel(r'$\dot{\theta}$',fontsize=font_size)
        plt.text(-15,-3,r'Control $u$',rotation=90,fontsize=font_size)
    if i == 3:
        plt.xlabel(r'$\theta$',fontsize=font_size) 
plt.colorbar()

plt.show()