import numpy as np
import math
import matplotlib.pyplot as plt
import torch
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
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out
    

hopf_model = Net(1,10,1)
hopf_model.load_state_dict(torch.load('./data/hopf/1d_hopf_net.pkl'))


m = 30
torch.manual_seed(10)  
rad = torch.Tensor(m,1).uniform_(3, 10)
ang = torch.Tensor(m,1).uniform_(0, 6.28)

def original_data(rad,ang,m,N=400,dt=0.001):
    X,W = torch.zeros([m,N]),torch.zeros([m,N])
    X1,X2 = torch.zeros([m,N]),torch.zeros([m,N])
    for r in range(m):
        X[r,0] = rad[r,0]
        W[r,0] = ang[r,0]
        for i in range(N-1):
            x = X[r,i]
            w = W[r,i]
            # u = hopf_model(torch.tensor([x-5.0]))
            new_x = x + x*(x-5.0)*(x+5.0)*dt 
            new_w = w + dt
            if new_x > 10.0:
                new_x = x 
                new_w = w
            X[r,i+1] = new_x
            W[r,i+1] = new_w
        X1[r,:]=X[r,:]*torch.cos(W[r,:])
        X2[r,:]=X[r,:]*torch.sin(W[r,:])
    orig_data = {'X':X,'W':W,'X1':X1,'X2':X2}
    torch.save(orig_data,'./data/hopf/data.pt')

def control_data(rad,ang,m=30,N=30000,dt=0.0001):
    start = timeit.default_timer()
    torch.manual_seed(9)  
    X,W = torch.zeros([m,N]),torch.zeros([m,N])
    X1,X2 = torch.zeros([m,N]),torch.zeros([m,N])
    # z = np.random.normal(0,1,N)
    
    for r in range(m):
        z = torch.randn(N)
        X[r,0] = rad[r,0]
        W[r,0] = ang[r,0]
        for i in range(N-1):
            x = X[r,i]
            w = W[r,i]
            u = hopf_model(torch.tensor([x-5.0]))
            new_x = x + x*(x-5.0)*(x+5.0)*dt + (x-5.0)*(u[0])*z[i]*math.sqrt(dt)
            new_w = w + dt
            X[r,i+1] = new_x
            W[r,i+1] = new_w
        X1[r,:]=X[r,:]*torch.cos(W[r,:])
        X2[r,:]=X[r,:]*torch.sin(W[r,:])
        print('{} done'.format(r))
    orig_data = {'X':X,'W':W,'X1':X1,'X2':X2}
    torch.save(orig_data,'./data/hopf/control_data.pt')
    stop = timeit.default_timer()
    print(stop-start)


def test():
    N = 100
    dt = 0.0001
    X = torch.zeros([1,N])
    W = torch.zeros([1,N])
    X[0,0] = 8.0
    W[0,0] = 3.8
    z = torch.randn(N)
    for i in range(N-1):
        x = X[0,i]
        w = W[0,i]
        u = hopf_model(torch.tensor([x-5.0]))
        new_x = x + x*(x-5.0)*(x+5.0)*dt + (x-5.0)*(u[0])*z[i]*math.sqrt(dt)
        new_w = w + dt
        X[0,i+1] = new_x
        W[0,i+1] = new_w
    X = X.clone().detach()
    plt.plot(np.arange(N),X[0,:],'r')
    plt.show()

if __name__ == '__main__':
    control_data(rad,ang,m,600,0.0001)
    original_data(rad,ang,m,400,0.001)
    test()
