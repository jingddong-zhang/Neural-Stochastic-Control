import torch 
import torch.nn.functional as F
import numpy as np
import timeit 
import math

class Net(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden,n_output)


    def forward(self,x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        out = self.layer2(h_1)
        return out


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
N = 1000             # sample size
D_in = 2        # input dimension
H1 = 6              # hidden dimension
D_out = 2           # output dimension
torch.manual_seed(10)  

x = torch.Tensor(N, D_in).uniform_(-10, 10)           

theta = 0.5
out_iters = 0
valid = False
while out_iters < 1 and not valid: 
    start = timeit.default_timer()

    model = Net(D_in,H1, D_out)

    i = 0 
    t = 0
    max_iters = 2000
    learning_rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []
    while i < max_iters and not valid: 
        out = model(x)

        g = out*x
        f = inverted_pendulum(x) 
        loss = (2-theta)*torch.diagonal(torch.mm(x,g.T))**2-torch.diagonal(torch.mm(x,x.T))*torch.diagonal(2*torch.mm(x,f.T)+torch.mm(g,g.T))
        # loss = (2-theta)*((x*g)**2)-x**2*(2*x*f+g**2)

        Lyapunov_risk = (F.relu(-loss)).mean()
        L.append(Lyapunov_risk)
        
        print(i, "Lyapunov Risk=",Lyapunov_risk.item()) 

        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step() 

        # if Lyapunov_risk == 0.0:
        #     break
        i += 1

    stop = timeit.default_timer()


    print('\n')
    print("Total time: ", stop - start)
    print("Verified time: ", t)
    
    out_iters+=1
    torch.save(torch.tensor(L), './data/inverted_pendulum/loss_AS.pt') 
    torch.save(model.state_dict(), './data/inverted_pendulum/algo2_invert_net.pkl') 