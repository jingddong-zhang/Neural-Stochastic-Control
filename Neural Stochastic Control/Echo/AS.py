import torch 
import torch.nn.functional as F
import numpy as np
import timeit 
import argparse
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--N', type=float, default=5000)
parser.add_argument('--lr', type=float, default=0.03)
args = parser.parse_args()

class Net(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(Net, self).__init__()
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



'''
For learning 
'''
N = args.N            # sample size
D_in = 50           # input dimension
H1 = 4*D_in             # hidden dimension
D_out = D_in         # output dimension
torch.manual_seed(10)  
x = torch.Tensor(N, D_in).uniform_(-10, 10)   

A = np.load('neural_sde/Echo/50/A_{}.npy'.format(D_in))
A = torch.tensor(A).to(torch.float32)




theta = 0.8
out_iters = 0
valid = False
while out_iters < 1 and not valid: 
    # break
    start = timeit.default_timer()

    model = Net(D_in,H1, D_out)

    i = 0 
    t = 0
    max_iters = 10000
    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    while i < max_iters and not valid: 
        out = model(x)
        g = out*x
        f = torch.relu(torch.mm(A,x.T)).T
        loss = (2-theta)*torch.diagonal(torch.mm(x,g.T))**2-torch.diagonal(torch.mm(x,x.T))*torch.diagonal(2*torch.mm(x,f.T)+torch.mm(g,g.T))
        # loss = (2-theta)*((x*g)**2)-x**2*(2*x*f+g**2)
        Lyapunov_risk = (F.relu(-loss)).mean()
        
        print(i, "Lyapunov Risk=",Lyapunov_risk.item()) 

        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step() 

        if Lyapunov_risk == 0:
            break
        i += 1

    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)
    print("Verified time: ", t)
    
    out_iters+=1

    
    torch.save(model.state_dict(), './data/Echo/AS_{}_relu_net.pkl'.format(D_in))