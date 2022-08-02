import torch
import torch.nn.functional as F
import numpy as np
import timeit
import argparse

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--N', type=float, default=1000)
parser.add_argument('--num', type=float, default=2)
parser.add_argument('--lr', type=float, default=0.05)
args = parser.parse_args()


class ControlNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(ControlNet, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, data):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(data))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        x = data
        return out * x

class Net(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self._scontrol = ControlNet(n_input,n_hidden,n_output)
        self._dcontrol = ControlNet(n_input,n_hidden,n_output)

    def forward(self,data):
        s_u = self._scontrol(data)
        d_u = self._dcontrol(data)
        return d_u,s_u


def f_(data,u):
    G = 9.81  # gravity
    L = 0.5  # length of the pole
    m = 0.15  # ball mass
    b = 0.1  # friction
    z = torch.zeros_like(data)
    for i in range(len(data)):
        x,y=data[i,:]
        z[i,:] = torch.tensor([y,G*np.sin(x)/L +(-b*y)/(m*L**2)])#+u[i]
    return z

def g_(data,u):
    z = torch.zeros_like(data)
    for i in range(len(data)):
        z[i,:] = 0.0+u[i]
    return z


'''
For learning 
'''
N = args.N  # sample size
D_in = 2  # input dimension
H1 = 3 * D_in  # hidden dimension
D_out = 2  # output dimension
torch.manual_seed(10)
Data = torch.Tensor(N,2).uniform_(-10,10)

theta = 0.8
out_iters = 0
while out_iters < 1:
    # break
    start = timeit.default_timer()
    model = Net(D_in, H1, D_out)
    i = 0
    t = 0
    max_iters = 200
    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    while i < max_iters:
        d_u,s_u = model(Data)
        f = f_(Data,d_u)
        g = g_(Data,s_u)

        x = Data
        # loss = (2-theta)*torch.diagonal(torch.mm(x, g.T))**2-torch.diagonal(torch.mm(x,x.T))*torch.diagonal(
        #     2*torch.mm(x,f.T)+torch.mm(g,g.T))
        loss = (2-theta)*((x*g)**2)-x**2*(2*x*f+g**2)
        # L_B = 2*(v-M/2)*f[:,3:4]/h(v)**2+g[:,3:4]**2/h(v)**2+4*g[:,3:4]**2*(v-M/2)**2/h(v)**3 - gamma*torch.log(1+torch.abs(h(v))) # barrier function 1
        # L_B = (2*(v-M/2)*f[:,3:4]/h(v)**2+g[:,3:4]**2/h(v)**2+4*g[:,3:4]**2*(v-M/2)**2/h(v)**3)
        # lossB = 2*L_B/h(v)-(1-theta)*(2*(v-M/2)*g[:,3:4])**2/h(v)**4
        AS_loss = (F.relu(-loss)).mean()
        print(i, "AS loss=", AS_loss.item())

        optimizer.zero_grad()
        AS_loss.backward()
        optimizer.step()

        if AS_loss < 1e-8:
            break
        # if AS_loss<0.5:
        #     optimizer=torch.optim.Adam(model.parameters(),lr=0.005)
        i += 1

    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)
    print("Verified time: ", t)

    out_iters += 1
    torch.save(model._scontrol.state_dict(),'./data/node_S.pkl')
# torch.save(model._dcontrol.state_dict(),'./data/D.pkl')
