from statistics import mean
import sys
sys.path.append('./neural_sde')
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import timeit 
# import pylustrator
# pylustrator.start()
start = timeit.default_timer()

A = torch.load('./neural_sde/hyper_a/data.pt')
A = A[:,-8:-1,:,:]
print(A.shape)

def plot_trajec(L,a):
    mean_data = torch.mean(L,0).detach().numpy()
    std_data  =torch.std(L,0).detach().numpy()
    plt.fill_between(np.arange(len(mean_data)),mean_data-std_data,mean_data+std_data,color='r',alpha=0.2)
    plt.plot(np.arange(len(mean_data)),mean_data,color='r',alpha=0.9,label=r'$b={}$'.format(a))
    plt.ylim(-1,6)
    # plt.xlabel('Time')
    plt.yticks([])
    plt.xticks([0.0, 6000], ["$0$", "$0.6$"])


