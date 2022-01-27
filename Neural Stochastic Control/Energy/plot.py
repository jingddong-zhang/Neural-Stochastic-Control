import numpy as np
import matplotlib.pyplot as plt
import torch

import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
matplotlib.rcParams['text.usetex'] = True

def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)
    pass

'''
Data corresponding to (a) in Figure 4, strength k from 0.2:10:0.2, 20 sample trajectories for each k, 
we choose dt=1e-5 and N=1e5 in Euler method. Data form is dictionary with key 'data' and 'end', the size 
for 'data' is [50,10001,20], 'end' corresponds to the average position over 20 trajectories for each k, the size is [50]
'''
Data = torch.load('./k_table_x0_20.pt')
data = Data['data']
endpoint = Data['end']
endpoint = torch.log(1+endpoint)
T = len(data)
dt = 0.00001
fontsize = 30

fig = plt.figure()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
fig1 = plt.subplot(141)
plt.scatter(np.arange(T) / 5,endpoint, s=45, c=endpoint, marker='.',alpha=0.85,cmap='rainbow')
plt.axvline(28/5,ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.5)
plt.ylabel(r'$\log(1+x)$', fontsize=fontsize)
plt.xlabel(r'$k$', fontsize=fontsize)
# cb = plt.colorbar()
# cb.set_ticks([0, 5, 10, 15])
# cb.ax.tick_params(labelsize=fontsize)
plt.xticks([0, 2, 4, 6, 8, 10],
           # ["0", "", "0.5", "","1.0", "", "1.5", "", "2.0"]
           )
plt.yticks([0, 5, 10, 15],
           # ["0", "", "0.5", "","1.0", "", "1.5", "", "2.0"]
           )
plot_grid()
plt.tick_params(labelsize=fontsize)


'''
Fix k=6，20 trajectories for linear control and neural stochastic control from initial 20.0，we set dt = 1e-5, N = 1e5 
in Euler method, the random seeds are set as 4*r+1 for r in range(20), the data form is dictionary with key 'data', the 
data size is [2,10001,20], data[0,:] corresponds to trajectories for learning control, data[1,:] corresponds to linear control.
'''

# Data = torch.load('./neural_sde/Energy/20seed_learning_control.pt')
Data = torch.load('./data/Energy/20seed_learning_control.pt')
data = Data['data']

fig2 = plt.subplot(154)
X = data[1,:]
X = X[:50000,:]
mean_data = torch.mean(X,1)
std_data = torch.std(X,1)


plt.fill_between(np.arange(len(X)) * dt,mean_data-std_data,mean_data+std_data,color='r',alpha=0.2)
plt.plot(np.arange(len(X)) * dt,mean_data,color='r',alpha=0.9,label='Linear control')
# plt.title('ME:{}'.format(38418))
plt.ylim([-100, 200])
plt.xlabel(r'Time', fontsize=fontsize)
plt.ylabel(r'$x$', fontsize=fontsize)
plt.xticks([0, 0.125, 0.25, 0.375, 0.5],
           ["$0$", "$~$","$0.25$","$~$", "$0.5$"]
           )
plt.yticks([-100, 0, 100, 200])
plt.legend(fontsize=fontsize * 0.5)
plot_grid()
plt.tick_params(labelsize=fontsize)


fig3 = plt.subplot(155)
Y = data[0,:]
Y = Y[:14000,:]
mean_data = torch.mean(Y,1)
std_data = torch.std(Y,1)
plt.fill_between(np.arange(len(Y))*dt,mean_data-std_data,mean_data+std_data,color='g',alpha=0.2)
plt.plot(np.arange(len(Y))*dt,mean_data,color='g',alpha=0.9,label='Learned control')
# plt.ylim([-100, 200])
plt.xlabel(r'Time', fontsize=fontsize)
plt.xticks([0, 0.075/2, 0.075, (0.075 + 0.15)/2, 0.15],
           ["$0$", "$~$","$0.075$", "$~$", "$0.15$"]
           )
plt.ylabel(r'$x$', fontsize=fontsize)
plt.yticks([-20, 0, 20, 40],
           # ["0", "0.05","0.1", "0.15"]
           )
plt.legend(fontsize=fontsize * 0.5)
plot_grid()
plt.tick_params(labelsize=fontsize)
# plt.title('ME:{}'.format(1375))

plt.show()