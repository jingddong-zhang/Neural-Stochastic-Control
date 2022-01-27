import torch 
import torch.nn.functional as F
import numpy as np
import timeit 

class Net(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden,n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        sigmoid = torch.nn.Tanh()
        # sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out
    

def f_value(x):
    y = []
    for i in range(0,len(x)):
        f = [x[i]*(x[i]+5)*(x[i]+10)]
        y.append(f)
    y = torch.tensor(y)
    return y


'''
For learning 
'''
N = 3000             # sample size
D_in = 1            # input dimension
H1 = 10              # hidden dimension
D_out = 1           # output dimension
torch.manual_seed(10)  

x = torch.Tensor(N, D_in).uniform_(-30, 30)           

theta = 0.5
out_iters = 0
while out_iters < 1: 
    start = timeit.default_timer()

    model = Net(D_in,H1, D_out)

    i = 0 
    t = 0
    max_iters = 700
    learning_rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []
    while i < max_iters: 
        out = model(x)

        g = out*x
        f = f_value(x) 
        # loss = (2-theta)*torch.diagonal(torch.mm(x,g.T))**2-torch.diagonal(torch.mm(x,x.T))*torch.diagonal(2*torch.mm(x,f.T)+torch.mm(g,g.T))
        loss = (2-theta)*((x*g)**2)-x**2*(2*x*f+g**2)
        Lyapunov_risk = (F.relu(-loss)).mean()
        L.append(Lyapunov_risk)
        
        print(i, "Lyapunov Risk=",Lyapunov_risk.item()) 

        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step() 

        i += 1

    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)
    print("Verified time: ", t)
    
    out_iters+=1
    # torch.save(torch.tensor(L), './data/hopf/loss_AS.pt') 
    # torch.save(model.state_dict(), './data/hopf/1d_hopf_net.pkl') 