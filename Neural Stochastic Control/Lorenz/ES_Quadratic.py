import torch.nn.functional as F
import timeit 
from hessian import hessian
from hessian import jacobian
# from gradient import hessian
# from gradient import jacobian
import torch
import random
import math
import numpy as np
def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(10)
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

class ControlNet(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(ControlNet, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden,n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out

class VNet(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(VNet, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden,n_output)

        
    def forward(self,x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out

class Net(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self._v = VNet(n_input,12,n_output)
        self._control = ControlNet(n_input,n_hidden,n_output)
        
    def forward(self,x):
        v = self._v(x)
        u = self._control(x)
        return v,u*x
    

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
# torch.manual_seed(10)
data_x = torch.Tensor(N, D_in).uniform_(0, 10)
# x = torch.Tensor(N, D_in).uniform_(-10, 10)
l = 0.001

start = timeit.default_timer()
model = Net(D_in,H1, D_out)
max_iters = 2000 
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for r in range(1, args.niters + 1):
    i = 0 
    L = []
    x = get_batch(data_x)
    while i < max_iters: 
        V_net, u = model(x)
        W1 = model._v.layer1.weight
        W2 = model._v.layer2.weight
        W3 = model._v.layer3.weight
        # W4 = model._v.layer4.weight
        B1 = model._v.layer1.bias
        B2 = model._v.layer2.bias
        B3 = model._v.layer3.bias
        # B4 = model._v.layer4.bias

        f = Lorenz_value(x)
        # f = modify_Lorenz_value(x)
        g = u 


        x = x.clone().detach().requires_grad_(True)
        output = torch.mm(F.tanh(torch.mm(F.tanh(torch.mm(x,W1.T)+B1),W2.T)+B2),W3.T)+B3
        # output = torch.mm(torch.tanh(torch.mm(x,W1.T)+B1),W2.T)+B2
        # V = torch.sum(output)
        num_v = torch.sum(l*x*x + ( x*output)**2,1)
        # num_v = torch.sum(output,1)
        V = torch.sum(l*x*x + (x*output)**2)
        Vx = jacobian(V,x)
        Vxx = hessian(V,x)
        loss = torch.zeros(N)

        for r in range(args.batch_size):
            L_V = torch.sum(Vx[0,3*r:3*r+3]*f[r,:]) + 0.5*torch.mm(g[r,:].unsqueeze(0),torch.mm(Vxx[3*r:3*r+3,3*r:3*r+3],g[r,:].unsqueeze(1)))
            Vxg = torch.sum(Vx[0,3*r:3*r+3]*g[r,:])
            v = num_v[r]
            loss[r] = Vxg**2/(v**2) - args.b*L_V/v
        
        Lyapunov_risk = (F.relu(-loss)).mean()
        L.append(Lyapunov_risk.item())
        print(i, "Lyapunov Risk=",Lyapunov_risk.item()) 

        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step() 
        if Lyapunov_risk < 1.0:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        elif Lyapunov_risk > 1.0:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if Lyapunov_risk == 0.0:
            break
        i += 1


    stop = timeit.default_timer()

    
    print('\n')
    print("Total time: ", stop - start)
    # torch.save(model._control.state_dict(),'ES_net.pkl')
    # torch.save(model._v.state_dict(),'ES_V_net.pkl')
# torch.save(model._control.state_dict(),'./data/Lorenz/ES_quad_net_modify_1.pkl')
# torch.save(model._v.state_dict(),'./data/Lorenz/ES_quad_V_net_modify_1.pkl')