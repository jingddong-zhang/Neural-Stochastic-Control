import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.5, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.5, ls='-', lw=1)

'''
Calculate and plot the mean end position of trajectories under learning control with each $\alpha$ 
'''
A = torch.load('./data/hyper_a/data.pt')
A = A[:,:-1,:,:]
print(A.shape)
end = torch.zeros([19])
for r in range(19):
    end[r] = torch.mean(A[0,r,:,-1])

print(end.shape)
end = end.detach().numpy()
plt.scatter(np.arange(len(end)),end, s=45, c=end, marker='.',alpha=0.99,cmap='rainbow')
plot_grid()
# plt.axvline(7.5,ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.3) 
plt.axvline(11.5,ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.3) 
plt.axhline(0.0,ls="--",linewidth=2.5,color="#dc8ff6",alpha=0.3) 
plt.yticks([0,0.03,0.06])
plt.ylabel(r'$\theta$')    
plt.xlabel(r'$\alpha$') 
plt.colorbar()

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(12.040000/2.54, 5.670000/2.54, forward=True)
plt.figure(1).ax_dict["<colorbar>"].set_position([0.895507, 0.226426, 0.016383, 0.696457])
plt.figure(1).axes[0].set_xlim(-1.0, 18.9)
plt.figure(1).axes[0].set_xticks([-1.0,  3.0, 7.0, 11.0, 15.0, 19.0])
plt.figure(1).axes[0].set_xticklabels(["0", "0.2",  "0.4",  "0.6",  "0.8", "1.0"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="center")
plt.figure(1).axes[0].set_position([0.139423, 0.226426, 0.739233, 0.696457])
plt.figure(1).axes[0].get_xaxis().get_label().set_fontsize(12)
plt.figure(1).axes[0].get_yaxis().get_label().set_fontsize(12)
#% end: automatic generated code from pylustrator
plt.show()
