import torch 
import torch.nn.functional as F
import timeit 
from hessian import hessian
from hessian import jacobian


class Net(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden,n_output)
        self.control = torch.nn.Linear(n_input,2,bias=False)


    def forward(self,x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = self.layer2(h_1)
        u = self.control(x)
        return out,u
    
def harmonic(x):
    y = []
    beta = 0.5
    for i in range(0,len(x)):
        f = [x[i,1],-x[i,0]-2*beta*x[i,1]]
        y.append(f)
    y = torch.tensor(y)
    return y

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
l = 0.01
x_0 = torch.zeros([1,2])
out_iters = 0
# valid = False
while out_iters < 1: 
    start = timeit.default_timer()
    model = Net(D_in,H1, D_out)

    i = 0 
    max_iters = 200 
    learning_rate = 0.03
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []
    while i < max_iters: 
        # start = timeit.default_timer()
        V_net, u = model(x)
        W1 = model.layer1.weight
        W2 = model.layer2.weight
        B1 = model.layer1.bias
        B2 = model.layer2.bias
        X0,u0 = model(x_0)

        f = harmonic(x)
        g = harmonic_control(x,u)


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
        q = model.control.weight.data.numpy()
    
        i += 1
    print(q)

    stop = timeit.default_timer()

    
    print('\n')
    print("Total time: ", stop - start)
    
    out_iters+=1
    # torch.save(torch.tensor(L),'./data/harmonic/loss_quad.pt')