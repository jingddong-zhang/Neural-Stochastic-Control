import torch 
import torch.nn.functional as F
import timeit 

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
# Drift function
def harmonic(x):
    y = []
    beta = 0.5
    for i in range(0,len(x)):
        f = [x[i,1],-x[i,0]-2*beta*x[i,1]]
        y.append(f)
    y = torch.tensor(y)
    return y
# Add control
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
x_0 = torch.zeros_like(x)

theta = 0.75
out_iters = 0
while out_iters < 1: 
    # break
    start = timeit.default_timer()

    model = Net(D_in,H1, D_out)

    i = 0 
    t = 0
    max_iters = 200
    learning_rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = []
    while i < max_iters: 
        # start = timeit.default_timer()
        out = model(x)
        u = out*x
        f = harmonic(x)
        g = harmonic_control(x,u)
        # Both loss are efficient
        # loss = (2-theta)*torch.diagonal(torch.mm(x,g.T))**2-torch.diagonal(torch.mm(x,x.T))*torch.diagonal(2*torch.mm(x,f.T)+torch.mm(g,g.T))
        loss = (2-theta)*((x*g)**2)-x**2*(2*x*f+g**2) 
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
        if Lyapunov_risk == 0:
            break
        # stop = timeit.default_timer()
        # print('per :', stop-start)
        i += 1

    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)
    
    out_iters+=1

    # torch.save(torch.tensor(L), './data/harmonic/loss_AS.pt') 
    # torch.save(model.state_dict(), './data/harmonic/algo2_net.pkl') 