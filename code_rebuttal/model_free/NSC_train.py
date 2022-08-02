import torch
import torch.nn.functional as F
import numpy as np
import timeit
import argparse

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--N', type=float, default=1000)
parser.add_argument('--num', type=float, default=6)
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
        data = data[:,1:2]
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(data))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        x = data
        # return out*x*torch.tensor([0.0,1.0,1.0,0.0,0.0,0.0])
        return out * x

class Net(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self._scontrol = ControlNet(n_input,n_hidden,n_output)
        # self._dcontrol = ControlNet(n_input,n_hidden,n_output)

    def forward(self,data):
        s_u = self._scontrol(data)
        # d_u = self._dcontrol(data)
        return s_u

def f_(data):
    a, b, c = 1, 1, 1
    z = torch.zeros_like(data)
    U2 = torch.tensor([[0.5, 0.74645887, 1.05370735, 0.38154169, 1.68833014, 0.83746371]])
    x = data + U2
    for i in range(len(data)):
        x1, x2, x3, x4, x5, x6 = x[i,:]
        z[i, 0] = 0.5 - a * x1
        z[i, 1] = 5 * x1 / ((1 + x1) * (1 + x3 ** 4)) - b * x2
        z[i, 2] = 5 * x4 / ((1 + x4) * (1 + x2 ** 4)) - c * x3
        z[i, 3] = 0.5 / (1 + x2 ** 4) - a * x4
        z[i, 4] = (x1 * x4 / (1 + x1 * x4) + 4 * x3 / (1 + x3)) / (1 + x2 ** 4) - a * x5
        z[i, 5] = (x1 * x4 / (1 + x1 * x4) + 4 * x2 / (1 + x2)) / (1 + x3 ** 4) - a * x6
        # x,y=data[i,:]
        # z[i,:] = torch.tensor([y,G*np.sin(x)/L +(-b*y)/(m*L**2)])#+u[i]
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
D_in = 1  # input dimension
H1 = 6 * D_in  # hidden dimension
D_out = 1  # output dimension
torch.manual_seed(10)
# Data = torch.Tensor(N,6).uniform_(-5,5)
Data = torch.load('./data/node1.pt')
# print(Data.shape)

theta = 0.9
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
        s_u = model(Data)
        f = f_(Data)[:,1:2]
        # g = g_(Data,s_u)[:,1:3]
        g = s_u
        x = Data[:,1:2]
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
    torch.save(model._scontrol.state_dict(),'./data/node_S_2.pkl')
# torch.save(model._dcontrol.state_dict(),'./data/D.pkl')
