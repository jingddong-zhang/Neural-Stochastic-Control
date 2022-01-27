import torch
import torch.nn as nn
import torch.nn.functional as F

class ICNN(nn.Module):
    def __init__(self, input_shape, layer_sizes, activation_fn):
        super(ICNN, self).__init__()
        self._input_shape = input_shape
        self._layer_sizes = layer_sizes
        self._activation_fn = activation_fn
        ws = []
        bs = []
        us = []
        prev_layer = input_shape
        w = torch.empty(layer_sizes[0], *input_shape)
        nn.init.xavier_normal_(w)
        ws.append(nn.Parameter(w))
        b = torch.empty([layer_sizes[0], 1])
        nn.init.xavier_normal_(b)
        bs.append(nn.Parameter(b))
        for i in range(len(layer_sizes))[1:]:
            w = torch.empty(layer_sizes[i], *input_shape)
            nn.init.xavier_normal_(w)
            ws.append(nn.Parameter(w))
            b = torch.empty([layer_sizes[i], 1])
            nn.init.xavier_normal_(b)
            bs.append(nn.Parameter(b))
            u = torch.empty([layer_sizes[i], layer_sizes[i-1]])
            nn.init.xavier_normal_(u)
            us.append(nn.Parameter(u))
        self._ws = nn.ParameterList(ws)
        self._bs = nn.ParameterList(bs)
        self._us = nn.ParameterList(us)

    def forward(self, x):
        # x: [batch, data]
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        else:
            data_dims = list(range(1, len(self._input_shape) + 1))
            x = x.permute(*data_dims, 0)
        z = self._activation_fn(torch.addmm(self._bs[0], self._ws[0], x))
        for i in range(len(self._us)):
            u = F.softplus(self._us[i])
            w = self._ws[i + 1]
            b = self._bs[i + 1]
            z = self._activation_fn(torch.addmm(b, w, x) + torch.mm(u, z))
        return z

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
    

class LyapunovFunction(nn.Module):
    def __init__(self,n_input,n_hidden,n_output,input_shape,smooth_relu_thresh=0.1,layer_sizes=[64, 64],lr=3e-4,eps=1e-3):
        super(LyapunovFunction, self).__init__()
        torch.manual_seed(2)
        self._d = smooth_relu_thresh
        self._icnn = ICNN(input_shape, layer_sizes, self.smooth_relu)
        self._eps = eps
        self._control = ControlNet(n_input,n_hidden,n_output)
   

    def forward(self, x):
        g = self._icnn(x)
        g0 = self._icnn(torch.zeros_like(x))
        u = self._control(x)
        u0 = self._control(torch.zeros_like(x))
        return self.smooth_relu(g - g0) + self._eps * x.pow(2).sum(dim=1), u*x 
        # return self.smooth_relu(g - g0) + self._eps * x.pow(2).sum(dim=1), u-u0 

    def smooth_relu(self, x):
        relu = x.relu()
        # TODO: Is there a clean way to avoid computing both of these on all elements?
        sq = (2*self._d*relu.pow(3) -relu.pow(4)) / (2 * self._d**3)
        lin = x - self._d/2
        return torch.where(relu < self._d, sq, lin)

def lya(ws,bs,us,smooth,x,input_shape):
    if len(x.shape) < 2:
        x = x.unsqueeze(0)
    else:
        data_dims = list(range(1, len(input_shape) + 1))
        x = x.permute(*data_dims, 0)
    z = smooth(torch.addmm(bs[0],ws[0], x))
    for i in range(len(us)):
        u = F.softplus(us[i])
        w = ws[i + 1]
        b = bs[i + 1]
        z = smooth(torch.addmm(b, w, x) + torch.mm(u, z))
    return z

