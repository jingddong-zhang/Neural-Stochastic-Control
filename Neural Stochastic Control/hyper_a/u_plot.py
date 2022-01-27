import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import cm
import matplotlib as mpl


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
        return out*x
D_in = 2          
H1 = 6             
D_out = 2  

model = ControlNet(D_in,H1,D_out)
vnorm = mpl.colors.Normalize(vmin=-80, vmax=80)

def draw_image2(f):
    with torch.no_grad():  
        x = torch.linspace(-6, 6, 200)
        y = torch.linspace(-6, 6, 200)
        X, Y = torch.meshgrid(x, y)
        inp = torch.stack([X, Y], dim=2)
        image = f(inp) 
        image = image[..., 0].detach().cpu() 

    plt.imshow(image, extent=[-6, 6, -6, 6], cmap='rainbow',norm=vnorm)
    # plt.xlabel(r'$\theta$')
    plt.xticks([-6,0,6])
    plt.yticks([])
    return image

def draw(a):
    model.load_state_dict(torch.load('./neural_sde/hyper_a/a_{}.pkl'.format(a)))
    draw_image2(model)
