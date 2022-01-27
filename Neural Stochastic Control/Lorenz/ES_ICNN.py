import torch.nn.functional as F
import timeit 
from hessian import hessian
from hessian import jacobian
# from gradient import hessian
# from gradient import jacobian

import torch
import random
import numpy as np
def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(10)

from Control_Nonlinear_Icnn import *
import math 

import argparse
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--N', type=int, default=10000)
parser.add_argument('--D_in', type=int, default=3)
parser.add_argument('--D_h', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.03)
parser.add_argument('--b', type=float, default=2.1)
parser.add_argument('--niters', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()

def Lorenz_value(x):
    y = []
    for i in range(0,len(x)):
        x1,x2,x3 = x[i,0],x[i,1],x[i,2]
        f = [10*(x2-x1),x1*(28-x3)-x2,x1*x2-x3*8/3]
        y.append(f)
    y = torch.tensor(y)
    return y

def modify_Lorenz_value(x):
    y = []
    e = torch.tensor([6.0*math.sqrt(2), 6.0*math.sqrt(2) , 27.0])
    for i in range(0,len(x)):
        x1,x2,x3 = x[i,:] + e
        f = [10*(x2-x1),x1*(28-x3)-x2,x1*x2-x3*8/3]
        y.append(f)
    y = torch.tensor(y)
    return y

def get_batch(data):
    s = torch.from_numpy(np.random.choice(np.arange(args.N, dtype=np.int64), args.batch_size, replace=False))
    batch_x = data[s,:]  # (M, D)
    return batch_x

'''
For learning 
'''
N = args.N          # sample size
D_in = args.D_in            # input dimension
H1 =  args.D_h             # hidden dimension
D_out = D_in          # output dimension

data_x = torch.Tensor(N, D_in).uniform_(0, 10)
eps = 0.001
start = timeit.default_timer()
model = LyapunovFunction(D_in,H1,D_out,(D_in,),0.1,[12,12,12,1],eps)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
max_iters = 2000
for r in range(1, args.niters + 1):
    # break
    x = get_batch(data_x) 
    i = 0 

    L = []
    while i < max_iters: 
        output, u = model(x)

        g = u*x 
        f = Lorenz_value(x)
        # f = modify_Lorenz_value(x)
        x = x.clone().detach().requires_grad_(True)
        ws = model._icnn._ws
        bs = model._icnn._bs
        us = model._icnn._us
        smooth = model.smooth_relu
        input_shape = (D_in,)
        V1 = lya(ws,bs,us,smooth,x,input_shape)
        V0 = lya(ws,bs,us,smooth,torch.zeros_like(x),input_shape)
        num_V = smooth(V1-V0)+eps*x.pow(2).sum(dim=1)
        V = torch.sum(smooth(V1-V0)+eps*x.pow(2).sum(dim=1))

        Vx = jacobian(V,x)
        Vxx = hessian(V,x)
        loss = torch.zeros(N)
        for r in range(args.batch_size):
            L_V = torch.sum(Vx[0,D_in*r:D_in*r+D_in]*f[r,:]) + 0.5*torch.mm(g[r,:].unsqueeze(0),torch.mm(Vxx[D_in*r:D_in*r+D_in,D_in*r:D_in*r+D_in],g[r,:].unsqueeze(1)))
            Vxg = torch.sum(Vx[0,D_in*r:D_in*r+D_in]*g[r,:])
            v = num_V[0,r]
            loss[r] = Vxg**2/(v**2) - args.b*L_V/v
        
        Lyapunov_risk = (F.relu(-loss)).mean()
        L.append(Lyapunov_risk.item())
        print(i, "Lyapunov Risk=",Lyapunov_risk.item()) 

        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step() 
        if Lyapunov_risk < 1.0:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        elif Lyapunov_risk > 1.0:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if Lyapunov_risk == 0.0:
            print(Lyapunov_risk)
            break
        i += 1
        # torch.save(model._control.state_dict(),'ES_icnn_net.pkl')
        # torch.save(model._icnn.state_dict(),'ES_icnn_V_net.pkl')


stop = timeit.default_timer()
print('\n')
print("Total time: ", stop - start)

# torch.save(model._control.state_dict(),'ES_icnn_net.pkl')
# torch.save(model._icnn.state_dict(),'ES_icnn_V_net.pkl')
# torch.save(model._control.state_dict(),'./neural_sde/Lorenz/ES_icnn_net_modify_1.pkl')
# torch.save(model._icnn.state_dict(),'./neural_sde/Lorenz/ES_icnn_V_net_modify_1.pkl')