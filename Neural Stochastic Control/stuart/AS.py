import torch 
import torch.nn.functional as F
import numpy as np
import timeit 

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
    

'''
For learning 
'''

n = 20         
D_in = 2*n-1           # input dimension
H1 = 4*n              # hidden dimension
D_out = 2*n-1           # output dimension

Data = torch.load('./data/stuart/20_train_data_small.pt')
# Data = torch.load('./data/stuart/20_train_data.pt')
x = Data['X']
f = Data['Y']
print(x[:,20:])
theta = 0.75
out_iters = 0

valid=True
while out_iters < 1 and valid == True: 
    # break
    start = timeit.default_timer()

    model = Net(D_in,H1, D_out)

    i = 0 
    t = 0
    max_iters = 1000
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    L = torch.zeros(1000)
    while i < max_iters: 
        out = model(x)

        g = out*x
        loss = (2-theta)*((x*g)**2)-x**2*(2*x*f+g**2)
        Lyapunov_risk = (F.relu(-loss)).mean()
        # Lyapunov_risk.requires_grad_(True)
        
        print(i, "Lyapunov Risk=",Lyapunov_risk.item()) 

        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step() 
        L[i] = Lyapunov_risk
        i += 1

    stop = timeit.default_timer()
    
    print('\n')
    print("Total time: ", stop - start)
    print("Verified time: ", t)
    
    out_iters+=1
    torch.save({'loss':L},'./data/stuart/loss.pt')
    # torch.save(model.state_dict(), './neural_sde/stuart/n_20/20_net_small.pkl') 


