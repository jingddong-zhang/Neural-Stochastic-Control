import torch 
import torch.nn.functional as F
import timeit 
from hessian import hessian
from hessian import jacobian
from Control_Nonlinear_Icnn import *


# Drift function
def harmonic(x):
    y = []
    beta = 0.5
    for i in range(0,len(x)):
        f = [x[i,1],-x[i,0]-2*beta*x[i,1]]
        y.append(f)
    y = torch.tensor(y)
    return y

# Add stochastic control
def harmonic_control(x,u):
    y = []
    k1,k2 = -3,2.15
    for i in range(0,len(x)):
        f = [0.0,k1*x[i,0]+k2*x[i,1]]
        y.append(f)
    y = torch.tensor(y)
    y[:,0] = y[:,0] + u[:,0]
    y[:,1] = y[:,1] + u[:,1]
    return y
    

'''
For learning 
'''
N = 500             # sample size
D_in = 2            # input dimension
H1 = 6             # hidden dimension
D_out = 2           # output dimension
torch.manual_seed(10)  
x = torch.Tensor(N, D_in).uniform_(-6, 6)           

eps = 0.001
out_iters = 0
while out_iters < 1: 
    # break
    start = timeit.default_timer()
    model = LyapunovFunction(D_in,H1,D_out,(D_in,),0.1,[6,6,1],eps)
    i = 0 
    t = 0 
    max_iters = 200
    learning_rate = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []
    while i < max_iters: 
        # start = timeit.default_timer()
        output, u = model(x)
 
        f = harmonic(x)
        g = harmonic_control(x,u)


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

        for r in range(N):
            L_V = torch.sum(Vx[0,2*r:2*r+2]*f[r,:]) + 0.5*torch.mm(g[r,:].unsqueeze(0),torch.mm(Vxx[2*r:2*r+2,2*r:2*r+2],g[r,:].unsqueeze(1)))
            Vxg = torch.sum(Vx[0,2*r:2*r+2]*g[r,:])
            v = num_V[0,r]
            loss[r] = Vxg**2/(v**2) - 2.1*L_V/v
        
        Lyapunov_risk = (F.relu(-loss)).mean()
        L.append(Lyapunov_risk)
        print(i, "Lyapunov Risk=",Lyapunov_risk.item()) 

        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step() 

        if Lyapunov_risk < 2.0:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        if Lyapunov_risk < 0.001:
            break
        # stop = timeit.default_timer()
        # print('per:',stop-start)
        i += 1
    # torch.save(torch.tensor(L),'./data/harmonic/loss_icnn.pt')
    # torch.save(model._control.state_dict(),'./data/harmonic/icnn_net.pkl')
    stop = timeit.default_timer()


    print('\n')
    print("Total time: ", stop - start)
    
    out_iters+=1
