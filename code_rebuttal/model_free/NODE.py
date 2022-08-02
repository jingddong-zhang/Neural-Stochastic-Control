# import sys
# sys.path.append('./neural_sde/NODE')

import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
true_y0 = torch.tensor([[0.5, -0.9, 0.6, -0.6, -0.9, 0.5]]).to(device)
# true_y0 = torch.Tensor(10,6).uniform_(-2,2).to(device)
t = torch.linspace(0., 15., args.data_size).to(device)


class Cell_Fate_ODEFunc(nn.Module):
    dim = 6
    a, b, c = 1, 1, 1

    def forward(self, t, x):
        # x shape: [1, 6]
        dx = torch.zeros_like(x)
        U2 = torch.tensor([[0.5, 0.74645887, 1.05370735, 0.38154169, 1.68833014, 0.83746371]])
        x = x + U2
        x1, x2, x3, x4, x5, x6 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        dx[:, 0] = 0.5 - self.a * x1
        dx[:, 1] = 5 * x1 / ((1 + x1) * (1 + x3**4)) - self.b * x2
        dx[:, 2] = 5 * x4 / ((1 + x4) * (1 + x2**4)) - self.c * x3
        dx[:, 3] = 0.5 / (1 + x2**4) - self.a * x4
        dx[:, 4] = (x1 * x4 / (1 + x1 * x4) + 4 * x3 / (1 + x3)) / (1 + x2**4) - self.a * x5
        dx[:, 5] = (x1 * x4 / (1 + x1 * x4) + 4 * x2 / (1 + x2)) / (1 + x3**4) - self.a * x6
        return dx


with torch.no_grad():
    # true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    true_y = odeint(Cell_Fate_ODEFunc(), true_y0, t)


def get_batch():
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 50),
            nn.Tanh(),
            nn.Linear(50, 6),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)

    # optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    optimizer = optim.Adam(func.parameters(), lr=1e-2)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        print(itr, loss)
        # if itr % args.test_freq == 0:
        #     with torch.no_grad():
        #         pred_y = odeint(func, true_y0, t)
        #         loss = torch.mean(torch.abs(pred_y - true_y))
        #         print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
        #         visualize(true_y, pred_y, func, ii)
        #         ii += 1
        # torch.save(func.state_dict(),'./neural_sde/NODE/symmetry.pkl')
        end = time.time()

data = func(1.0, true_y)
torch.save(data[:,0,:], './data/node1.pt')
print(data.shape)