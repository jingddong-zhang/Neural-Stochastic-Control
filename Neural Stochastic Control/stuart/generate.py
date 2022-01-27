import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import math
import timeit 
from scipy.integrate import odeint
import sys
sys.path.append('./neural_sde/stuart')
from AS import *
from functions import *

start = timeit.default_timer()


stuart_model = Net(D_in,H1,D_out)
# stuart_model.load_state_dict(torch.load('./neural_sde/stuart/n_20/20_net.pkl'))
stuart_model.load_state_dict(torch.load('./data/stuart/20_net_small.pkl'))
torch.manual_seed(6) 
n = 20
L = torch.eye(n)-torch.ones([n,n])/n
N = 60000
dt = 0.0001
x0 = torch.cat([torch.Tensor(n).uniform_(0, 5),torch.Tensor(n-1).uniform_(-1.0,1.0)],0)   
R = x0[:20]
dW = x0[20:]


def original_20():
    # W = theta(dW)
    # x0 = torch.cat([R-1,W],0)
    X = torch.load('./data/stuart/20_original_data.pt')
    X = X['X']
    x0 = X[-1]
    X = torch.zeros(N+1,2*n)
    X[0,:] = x0
    for i in range(N):
        x = X[i,:]
        dx = original_f_value(x,L)
        new_x = x + dx*dt
        X[i+1,:]=new_x
        if i%100 == 0:
            print(i)
    torch.save({'X':X},'./data/stuart/20_original_data_add.pt')


def test():
    torch.manual_seed(7) 
    X = torch.load('./data/stuart/20_test_data_try.pt')
    X = X['X']
    x0 = X[-1]
    length = len(X)-1
    # length = 0

    # x0 = torch.cat([torch.Tensor(n).uniform_(0, 5),torch.Tensor(n-1).uniform_(-1.0,1.0)],0)   
    X = torch.zeros(N+1,2*n-1)
    X[0,:] = x0
    z = torch.randn(length+N,2*n-1)[length:,:]
    for i in range(N):
        x = X[i,:]
        with torch.no_grad():
            u = stuart_model(x)
        dx = f_value(x,L)
        new_x = x + dx*dt + x*u*z[i,:]*math.sqrt(dt)
        X[i+1,:]=new_x 
        if i%100 == 0:
            print(i)
    
    torch.save({'X':X},'./data/stuart/20_test_data_try_add.pt')

if __name__ == '__main__':
    original_20()           
    # test()
    stop = timeit.default_timer()
    print(stop-start)
