import numpy as np
import matplotlib.pyplot as plt
from V_plot import *
from u_plot import *
from plot_trajectory import *
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
# matplotlib.rcParams['text.usetex'] = True
font_size = 15

A = torch.load('./data/hyper_b/data.pt')[:,9:14,:,:] #pick trajectories correspond to 1.9,2.0,2.1,2.2,2.3 
# print(A.shape)

def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)

def plot_b(b):
    L = np.load('./data/hyper_b/b_{}.npy'.format(b))
    r_L =  np.zeros(1000-len(L))
    L = np.concatenate((L,r_L),axis=0)
    #  np.concatenate((a,b),axis=0)
    plt.plot(np.arange(len(L)),L,'b')
    plt.ylim(-1.0,25)
    plt.title('b = {}'.format(b))
    plt.xticks([0,400,800])
    plt.yticks([])



for i in range(5):
    plt.subplot(4, 5, i+1)

    plot_b(1.9+i*0.1)
    
    plot_grid()
    if i == 0:
        plt.yticks([0,10,20])
        plt.ylabel('Loss',fontsize=font_size)
        plt.text(-5,5,'Training',rotation=90,fontsize=font_size)
    else:
        plt.yticks([0, 10, 20], ['', '', ''])
    if i == 2:
        plt.xlabel('Iterations',fontsize=font_size) 


for i in range(5):
    plt.subplot(4, 5, 5 + i+1)
    plot_trajec(A[0,i,:,0:10000:10],1.9+i*0.1)
    plot_grid()
    if i == 0:
        plt.yticks([-10,-5,0,5,10])
        plt.ylabel(r'$\theta$',fontsize=font_size)
        plt.text(-1,-5,'Trajectory',rotation=90,fontsize=font_size)
    else:
        plt.yticks([-10,-5, 0,5, 10], ['', '', '','',''])
    if i == 2:
        plt.xlabel('Time',fontsize=font_size) 

for i in range(5):
    plt.subplot(4, 5, 10 + i+1)
    drawV(1.9+i*0.1)
    if i == 0:
        plt.yticks([-5,0,5])
        plt.ylabel(r'$\dot{\theta}$',fontsize=font_size)
        plt.text(-15,-5,'Lyapunov V',rotation=90,fontsize=font_size)
    if i == 2:
        plt.xlabel(r'$\theta$',fontsize=font_size) 
plt.colorbar()



for i in range(5):
    plt.subplot(4, 5, 15 + i+1)
    draw(1.9+i*0.1)
    if i == 0:
        plt.yticks([-5,0,5])
        plt.ylabel(r'$\dot{\theta}$',fontsize=font_size)
        plt.text(-15,-3,'Control u',rotation=90,fontsize=font_size)
    if i == 2:
        plt.xlabel(r'$\theta$',fontsize=font_size) 
plt.colorbar()

plt.show()