import sys
sys.path.append('./neural_sde')
import torch 
import torch.nn.functional as F
import numpy as np
import timeit 
from hessian import hessian
from hessian import jacobian
# from gradient import hessian
# from gradient import jacobian

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
        self.layer2 = torch.nn.Linear(n_hidden,n_output)

        
    def forward(self,x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = self.layer2(h_1)
        return out

class Net(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self._v = VNet(n_input,n_hidden,n_output)
        self._control = ControlNet(n_input,n_hidden,n_output)
        
    def forward(self,x):
        v = self._v(x)
        u = self._control(x)
        return v,u*x
    
def inverted_pendulum(x):
    y = []
    G = 9.81  # gravity
    L = 0.5   # length of the pole 
    m = 0.15  # ball mass
    b = 0.1   # friction
    for i in range(0,len(x)):
        f = [x[i,1],G*torch.sin(x[i,0])/L +(-b*x[i,1])/(m*L**2)]
        y.append(f)
    y = torch.tensor(y)
    return y

'''
For learning 
'''
N = 500             # sample size
D_in = 2            # input dimension
H1 = 6             # hidden dimension
D_out = 2           # output dimension
torch.manual_seed(10)  
x = torch.Tensor(N, D_in).uniform_(-10, 10)           
l = 0.01
# valid = False
# while out_iters < 1: 
for r in range(1):
    b = float(format(2.1 + r*0.1,'.1f'))
    start = timeit.default_timer()
    model = Net(D_in,H1, D_out)
    i = 0 
    t = 0 
    max_iters = 1000
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []
    while i < max_iters: 
        V_net, u = model(x)
        W1 = model._v.layer1.weight
        W2 = model._v.layer2.weight
        B1 = model._v.layer1.bias
        B2 = model._v.layer2.bias

        f = inverted_pendulum(x)
        g = u


        x = x.clone().detach().requires_grad_(True)
        output = torch.mm(torch.tanh(torch.mm(x,W1.T)+B1),W2.T)+B2
        # V = torch.sum(output)
        num_v = torch.sum(l*x*x + ( x*output)**2,1)
        # num_v = torch.sum(output,1)
        V = torch.sum(l*x*x + (x*output)**2)
        Vx = jacobian(V,x)
        Vxx = hessian(V,x)
        loss = torch.zeros(N)

        for r in range(N):
            L_V = torch.sum(Vx[0,2*r:2*r+2]*f[r,:]) + 0.5*torch.mm(g[r,:].unsqueeze(0),torch.mm(Vxx[2*r:2*r+2,2*r:2*r+2],g[r,:].unsqueeze(1)))
            Vxg = torch.sum(Vx[0,2*r:2*r+2]*g[r,:])
            v = num_v[r]
            loss[r] = Vxg**2/(v**2) - b*L_V/v
        

        Lyapunov_risk = (F.relu(-loss)).mean()
        L.append(Lyapunov_risk.item())
        print(i, "Lyapunov Risk=",Lyapunov_risk.item()) 

        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step() 
        # if Lyapunov_risk < 0.12:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # else:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # print(q)
        # if Lyapunov_risk < 1.0:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # else:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
        if Lyapunov_risk == 0.0:
            break
        i += 1
    
    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)

    
    # np.save('./neural_sde/hyper_b/b_{}.npy'.format(b), L)
    # torch.save(model._control.state_dict(),'./neural_sde/hyper_b/b_{}.pkl'.format(b))