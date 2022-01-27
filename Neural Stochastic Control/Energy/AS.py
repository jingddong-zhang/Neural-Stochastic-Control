import torch 
import torch.nn.functional as F
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
    

def f_value(x):
    y = []
    for i in range(0,len(x)):
        f = [x[i]*math.log(1+abs(x[i]))]
        y.append(f)
    y = torch.tensor(y)
    return y


'''
For learning 
'''
N = 4000             # sample size
D_in = 1            # input dimension
H1 = 6              # hidden dimension
D_out = 1           # output dimension
torch.manual_seed(10)  

x = torch.Tensor(N, D_in).uniform_(0,50)           

theta = 0.9
out_iters = 0
while out_iters < 1: 
    start = timeit.default_timer()

    model = Net(D_in,H1, D_out)

    i = 0 
    t = 0
    max_iters = 100
    learning_rate = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    while i < max_iters: 
        out = model(x)

        g = out*x
        f = f_value(x) 
        loss = (2-theta)*((x*g)**2)-x**2*(2*x*f+g**2)
        Lyapunov_risk = (F.relu(-loss)).mean()
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
    torch.save(model.state_dict(), './theta0.9_1d_log_net.pkl') 