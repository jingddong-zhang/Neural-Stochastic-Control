import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import cm
import matplotlib as mpl
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
# matplotlib.rcParams['text.usetex'] = True


colors = [
    [233/256,	110/256, 236/256], # #e96eec
    # [0.6, 0.6, 0.2],  # olive
    # [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
    [255/255, 165/255, 0],
    # [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
    # [223/256,	73/256,	54/256], # #df4936
    [107/256,	161/256,255/256], # #6ba1ff
    [0.6, 0.4, 0.8], # amethyst
    [0.0, 0.0, 1.0], # ao
    [0.55, 0.71, 0.0], # applegreen
    # [0.4, 1.0, 0.0], # brightgreen
    [0.99, 0.76, 0.8], # bubblegum
    [0.93, 0.53, 0.18], # cadmiumorange
    [11/255, 132/255, 147/255], # deblue
    [204/255, 119/255, 34/255], # {ocra}
]
colors = np.array(colors)


l = 0.01
class VNet(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(VNet, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden,n_output)

        
    def forward(self,x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = self.layer2(h_1)
        return l*x*x + (x*out)**2

D_in = 2          
H1 = 6             
D_out = 2  

vmodel = VNet(D_in,H1,D_out)
V_vnorm = mpl.colors.Normalize(vmin=0, vmax=2.0)
D = 6
def draw_imageV(f):
    with torch.no_grad():  
        x = torch.linspace(-D, D, 200)
        y = torch.linspace(-D, D, 200)
        X, Y = torch.meshgrid(x, y)
        inp = torch.stack([X, Y], dim=2) 
        image = f(inp)  
        image = image[..., 0].detach().cpu()

    plt.contour(X,Y,image-0.05,0,linewidths=2, colors=colors[-3],linestyles='--')
    # plt.contourf(X,Y,image,8,alpha=0.3,cmap='turbo',norm=vnorm)

    plt.imshow(image, extent=[-6, 6, -6, 6], cmap='rainbow',norm=V_vnorm)
    plt.xticks([-5,0,5])
    plt.yticks([])
    return image

def drawV(a):
    vmodel.load_state_dict(torch.load('./neural_sde/hyper_b/V_b_{}.pkl'.format(a)))
    draw_imageV(vmodel)
    # plt.title(r'b$={}$'.format(a))
