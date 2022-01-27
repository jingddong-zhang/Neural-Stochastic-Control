from functions import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
# matplotlib.rcParams['text.usetex'] = True
font_size =  35
def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.5)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)


'''
Plot trajectories and orbits
'''
L = 20000
E = 50000

plt1 = plt.subplot(231)
X = torch.load('./data/stuart/20_original_data_cat.pt')
X = X['X'][L:E:10,:]
X = transform(20,X)
for i in range(20):
    plt.plot(np.arange(len(X[:,0])),X[:,i],color = plt.cm.Accent(i/45))
plt.xticks([0,1000,2000,3000],[0,1.0,2.0,3.0],fontsize=font_size)
plt.yticks([-1,0,1],fontsize=font_size)
plot_grid()
plt.title(r'$x$',fontsize=font_size)
plt.ylabel('Without Control',fontsize=font_size)

plt2 = plt.subplot(232)
for i in range(20):
    plt.plot(np.arange(len(X[:,0])),X[:,i+20],color = plt.cm.Accent(i/45))
plt.xticks([0,1000,2000,3000],[0,1.0,2.0,3.0],fontsize=font_size)
plt.title(r'$y$',fontsize=font_size)
plt.yticks([-1,0,1],fontsize=font_size)
plot_grid()

plt3 = plt.subplot(233) 
for i in range(20):
    plt.plot(X[:,i+0],X[:,i+20],color = plt.cm.Accent(i/45),label='{}'.format(i))
plt.xticks([-1,0,1],fontsize=font_size)
plt.yticks([-1,0,1],fontsize=font_size)
plt.xlabel(r"$x$",fontsize=font_size)
plt.ylabel(r'$y$',fontsize=font_size)
plot_grid()
plt.title('Orbit',fontsize=font_size)

plt4 = plt.subplot(234)
X = diff_to_orig(20,'./data/stuart/20_original_data_cat.pt','./neural_sde/stuart/n_20/20_test_data_cat.pt')[L:E:10,:]
X = transform(20,X)
for i in range(20):
    plt.plot(np.arange(len(X[:,0])),X[:,i],color = plt.cm.Accent(i/45))
plot_grid()
plt.ylabel('With Control',fontsize=font_size)
plt.xticks([0,1000,2000,3000],[0,1.0,2.0,3.0],fontsize=font_size)
plt.yticks([-1,0,1],fontsize=font_size)
plt.xlabel('Time',fontsize=font_size)



plt5 = plt.subplot(235)
for i in range(20):
    plt.plot(np.arange(len(X[:,0])),X[:,i+20],color = plt.cm.Accent(i/45))
plot_grid()
plt.xticks([0,1000,2000,3000],[0,1.0,2.0,3.0],fontsize=font_size)
plt.yticks([-1,0,1],fontsize=font_size)
plt.xlabel('Time',fontsize=font_size)


plt6 = plt.subplot(236) 
for i in range(20):
    plt.plot(X[:,i+0],X[:,i+20],color = plt.cm.Accent(i/45),label='{}'.format(i))
plt.xticks([-1,0,1],fontsize=font_size)
plt.yticks([-1,0,1],fontsize=font_size)
plt.xlabel(r"$x$",fontsize=font_size)
plt.ylabel(r'$y$',fontsize=font_size)
plot_grid()
plt.show()


'''
Plot loss function
'''

# loss = torch.load('./data/stuart/loss.pt')
# loss = loss['loss'].detach()
# loss = loss[:30]


# fig = plt.figure(figsize=(6,8))
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
# plt1 = plt.subplot(121)
# # loss = loss.detach().numpy()
# plt.plot(np.arange(len(loss)),loss)

# plt2=plt.subplot(122)
# loss = loss[10:30]
# # loss = loss.detach().numpy()
# plt.plot(np.arange(len(loss)),loss)
# plt.plot()
# #% start: automatic generated code from pylustrator
# plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
# import matplotlib as mpl
# plt.figure(1).set_size_inches(14.120000/2.54, 9.110000/2.54, forward=True)
# plt.figure(1).axes[0].set_position([0.109847, 0.124637, 0.880047, 0.838141])
# plt.figure(1).axes[0].get_xaxis().get_label().set_text("iterations")
# plt.figure(1).axes[0].get_yaxis().get_label().set_text("loss")
# plt.figure(1).axes[1].set_xlim(-0.9500000000000001, 20.0)
# plt.figure(1).axes[1].set_ylim(-0.09267258382915317, 1.9471967105529984)
# plt.figure(1).axes[1].set_xticks([0.0, 10.0, 20.0])
# plt.figure(1).axes[1].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
# plt.figure(1).axes[1].set_xticklabels(["10", "20", "30"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="DejaVu Sans", horizontalalignment="center")
# plt.figure(1).axes[1].set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00", "1.25", "1.50", "1.75"], fontsize=10)
# plt.figure(1).axes[1].set_position([0.610715, 0.504267, 0.336851, 0.396884])
# #% end: automatic generated code from pylustrator
# plt.show()