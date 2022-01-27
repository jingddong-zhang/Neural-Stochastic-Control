import matplotlib.pyplot as plt
import torch
import numpy as np
# import pylustrator
# pylustrator.start()

def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.5, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.5, ls='-', lw=1)
A = torch.load('./neural_sde/hyper_b/data.pt')
print(A.shape)
end = torch.zeros([20])
for r in range(20):
    end[r] = torch.mean(A[0,r,:,-1])

print(end)
end = end.detach().numpy()
plt.scatter(np.arange(len(end)),end, s=45, c=end, marker='.',alpha=0.99,cmap='rainbow')
plot_grid()
plt.yticks([0,1,2])
plt.xticks([0.0,  4.0, 8.0,  12.0,  16.0,  20.0],["1.0", "1.4", "1.8", "2.2",  "2.6", "3.0"])
plt.axvline(8.5,ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.3) 
plt.axvline(13.5,ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.3) 
plt.axhline(0.0,ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.3) 
plt.ylabel(r'$\theta$')    
plt.xlabel(r'$b$') 
plt.colorbar()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(11.360000/2.54, 4.990000/2.54, forward=True)
plt.figure(1).ax_dict["<colorbar>"].set_position([0.931942, 0.234718, 0.014887, 0.679046])
plt.figure(1).axes[0].set_xlim(-0.9, 20.0)
# plt.figure(1).axes[0].set_xticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
# plt.figure(1).axes[0].set_xticklabels(["1.0", "1.2", "1.4", "1.6", "1.8", "2.0", "2.2", "2.4", "2.6", "2.8", "3.0"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="center")
# plt.figure(1).axes[0].grid(False)
plt.figure(1).axes[0].set_position([0.092998, 0.225654, 0.826345, 0.697175])
#% end: automatic generated code from pylustrator
plt.show()
