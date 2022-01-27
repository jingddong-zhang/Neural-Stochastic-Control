import numpy as np
import torch
import math

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



D_in = 50           # input dimension
H1 = 4*D_in             # hidden dimension
D_out = D_in
A = np.load('./data/Echo/A_{}.npy'.format(D_in))
A = torch.tensor(A).to(torch.float32)

m = 10
N = 200000
dt = 0.000001
model = Net(D_in,H1,D_out)
x0 = torch.linspace(-2,2,50)


def tanh_generate(m,N,dt):
    model.load_state_dict(torch.load('./data/Echo/AS_50_net.pkl'))
    X = torch.zeros(m,N+1,D_in)
    for r in range(m):
        torch.manual_seed(6*r+6)
        z  = torch.randn(N)
        X[r,0,:] = x0 
        for i in range(N):
            x = X[r,i,:].unsqueeze(1)
            with torch.no_grad():
                u = model(X[r,i,:]).unsqueeze(1)
            new_x = x + torch.tanh(torch.mm(A,x))*dt + math.sqrt(dt)*z[i]*u*x
            X[r,i+1,:]=new_x[:,0]
        print(r)
    X = X.detach().numpy()
    np.save('./data/Echo/tanh_data.npy',X)

def relu_generate(m,N,dt):
    model = Net(D_in,100,D_out)
    model.load_state_dict(torch.load('./data/Echo/AS_50_relu_net.pkl'))
    X = torch.zeros(m,N+1,D_in)
    for r in range(m):
        torch.manual_seed(6*r+6)
        z  = torch.randn(N)
        X[r,0,:] = x0 
        for i in range(N):
            x = X[r,i,:].unsqueeze(1)
            with torch.no_grad():
                u = model(X[r,i,:]).unsqueeze(1)
            new_x = x + torch.relu(torch.mm(A,x))*dt + math.sqrt(dt)*z[i]*u*x
            X[r,i+1,:]=new_x[:,0]
        print(r)
    X = X.detach().numpy()
    np.save('./data/Echo/relu_data.npy',X)

tanh_generate(m,N,dt)
relu_generate(m,N,dt) 


