import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import timeit 
start = timeit.default_timer()




def plot_trajec(L,b):
    mean_data = torch.mean(L,0).detach().numpy()
    std_data  =torch.std(L,0).detach().numpy()
    plt.fill_between(np.arange(len(mean_data)),mean_data-std_data,mean_data+std_data,color='r',alpha=0.2)
    plt.plot(np.arange(len(mean_data)),mean_data,color='r',alpha=0.9,label=r'$b={}$'.format(b))
    plt.ylim(-10,10)
    # plt.xlabel('Time')
    plt.xticks([0.0, 500,  1000], ["$0$", "$0.5$",  "$1.0$"])
    plt.yticks([])
