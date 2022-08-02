import numpy as np
import matplotlib.pyplot as plt
import torch

# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
# matplotlib.rcParams['text.usetex'] = True

def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)
    pass



energy = np.load('./data/energy_list.npy')
dt = 0.00001*10
# dt = 0.0001
fontsize = 15
data = torch.load('./data/k_table_x0_20.pt')
print(data.shape)
for i in range(5):
    plt.subplot(1,6,i+1)
    k=(i+1)*2
    X=data[10*(i+1)-1,0:50000:10,:]
    mean_data = torch.mean(X,1)
    std_data = torch.std(X,1)
    plt.fill_between(np.arange(len(X)) * dt,mean_data-std_data,mean_data+std_data,color='r',alpha=0.2)
    plt.plot(np.arange(len(X)) * dt,mean_data,color='r',alpha=0.9,label='k={}'.format(k))
    # plt.title('ME:{}'.format(38418))
    plt.ylim([-100, 200])
    plt.xlabel(r'Time', fontsize=fontsize)
    if i == 0:
        plt.ylabel(r'$x$', fontsize=fontsize)
    plt.xticks([0, 0.125, 0.25, 0.375, 0.5],
               ["$0$", "$~$","$0.25$","$~$", "$0.5$"]
               )
    plt.yticks([-100, 0, 100, 200])
    plt.legend(fontsize=fontsize)
    plot_grid()
    plt.title('ME:{}'.format(int(energy[10*(i+1)-1])))
    plt.tick_params(labelsize=fontsize)

Data = torch.load('./data/20seed_learning_control.pt')
data = Data['data'].detach().numpy()


dt = 0.00001
fig3 = plt.subplot(166)
Y = data[0,:]
Y = Y[:14000,:]
mean_data = np.mean(Y,1)
std_data = np.std(Y,1)
plt.fill_between(np.arange(len(Y))*dt,mean_data-std_data,mean_data+std_data,color='g',alpha=0.2)
plt.plot(np.arange(len(Y))*dt,mean_data,color='g',alpha=0.9,label='Learned control')
# plt.ylim([-100, 200])
plt.xlabel(r'Time', fontsize=fontsize)
plt.xticks([0, 0.075/2, 0.075, (0.075 + 0.15)/2, 0.15],
           ["$0$", "$~$","$0.075$", "$~$", "$0.15$"]
           )
plt.ylabel(r'$x$', fontsize=fontsize)
plt.yticks([-20, 0, 20, 40],
           ["0", "0.05","0.1", "0.15"]
           )
plt.legend(fontsize=fontsize * 0.7)
plot_grid()
plt.tick_params(labelsize=fontsize)
plt.title('ME:{}'.format(1438))

plt.show()