import sys
sys.path.append('./neural_sde')
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import timeit 

A = torch.ones(2,100)
# B = torch.diagonal(A)
print(A[:,0:100:10].shape)