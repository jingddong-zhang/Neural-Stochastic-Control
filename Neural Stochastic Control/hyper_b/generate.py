import numpy as np
import math
import torch
import timeit 
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(10)

class ControlNet(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(ControlNet, self).__init__()
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
D_in = 2          
H1 = 6             
D_out = 2  

model = ControlNet(D_in,H1,D_out)
set_state0 = torch.tensor([[-5.0,5.0]])
# set_state0 = torch.tensor([[-5.0,5.0],[-3.0,4.0],[-1.0,3.0],[1.0,-3.0],[3.0,-4.0],[5.0,-5.0]])
def control_data(model,random_seed,set_state0,M=6,N=20000,dt=0.00001):
    start = timeit.default_timer()
    torch.manual_seed(random_seed)  
    X1,X2 = torch.zeros([M,N]),torch.zeros([M,N])
    for r in range(M):
        G = 9.81  # gravity
        L = 0.5   # length of the pole 
        m = 0.15  # ball mass
        b = 0.1
        z = torch.randn(N)
        X1[r,0] = set_state0[r,0]
        X2[r,0] = set_state0[r,1]
        for i in range(N-1):
            x1 = X1[r,i]
            x2 = X2[r,i]
            with torch.no_grad():
                u = model(torch.tensor([x1,x2]))
            new_x1 = x1 + x2*dt + x1*u[0]*z[i]*math.sqrt(dt)
            new_x2 = x2 + (G*math.sin(x1)/L - b*x2/(m*L**2))*dt + x2*u[1]*z[i]*math.sqrt(dt)
            X1[r,i+1] = new_x1
            X2[r,i+1] = new_x2
       
        print('{} done'.format(r))
    # data = {'X1':X1,'X2':X2}
    # torch.save(data,'./neural_sde/hyper_b/b_{}.pt'.format(b))
    stop = timeit.default_timer()
    print(stop-start)
    return X1,X2

'''
Generate trajectories under control with corresponding b
'''
if __name__ == '__main__':
    M = 5
    N = 20000
    data = torch.zeros([2,10,M,N])
    for r in range(10):
        b = 2.0 + r*0.1
        model.load_state_dict(torch.load('./data/hyper_b/b_{}.pkl'.format(b)))
        # X1,X2=torch.zeros([M,N]),torch.zeros([M,N])
        for i in range(M):
            x1,x2 = control_data(model,i*6,set_state0,1,N,0.0001)
            # X1[i,:] = x1[0,:]
            # X2[i,:] = x2[0,:]
            data[0,r,i,:] = x1[0,:]
            data[1,r,i,:] = x2[0,:]
            print('({},{})'.format(r,i))
    torch.save(data,'data.pt')
        


# model.load_state_dict(torch.load('./neural_sde/hyper_b/b_{}.pkl'.format(1.6)))
# X1,X2 = control_data(model,6,set_state0,1,30000,0.0001)
# X1 = X1.detach().numpy()[0,:]
# print(X1.shape)
# plt.plot(np.arange(len(X1)),X1)
# plt.show()