import numpy as np
import matplotlib.pyplot as plt
import torch

import pylustrator
pylustrator.start()
import seaborn as sns
sns.set_theme(style="white")

def plot_a(a):
    L = np.load('./neural_sde/hyper_a/a_{}.npy'.format(a))
    r_L =  np.zeros(1000-len(L))
    L = np.concatenate((L,r_L),axis=0)
    #  np.concatenate((a,b),axis=0)
    plt.plot(np.arange(len(L)),L,'b')
    # plt.xlabel('Iterations')
    plt.ylim(-0.01,1)
    plt.yticks([])
    plt.title(r'$\alpha={}$'.format(a))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.subplot(171)
plot_a(0.65)
plt.ylabel('Loss')
plt.yticks([0,0.25,0.5,0.75,1.0])
plt.subplot(172)
plot_a(0.7)

plt.subplot(173)
plot_a(0.75)

plt.subplot(174)
plot_a(0.8)

plt.subplot(175)
plot_a(0.85)

plt.subplot(176)
plot_a(0.9)

plt.subplot(177)
plot_a(0.95)


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(14.460000/2.54, 4.880000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.118581, 0.256900, 0.084156, 0.543710])
plt.figure(1).axes[1].set_position([0.244815, 0.256900, 0.084156, 0.543710])
plt.figure(1).axes[1].title.set_position([0.500000, 1.000000])
plt.figure(1).axes[2].set_position([0.371050, 0.256900, 0.084156, 0.543710])
plt.figure(1).axes[3].set_position([0.497285, 0.256900, 0.084156, 0.543710])
plt.figure(1).axes[4].set_position([0.623519, 0.256900, 0.084156, 0.543710])
plt.figure(1).axes[5].set_position([0.749754, 0.256900, 0.084156, 0.543710])
plt.figure(1).axes[6].set_position([0.875988, 0.256900, 0.084156, 0.543710])
plt.figure(1).text(0.5, 0.5, 'New Text', transform=plt.figure(1).transFigure)  # id=plt.figure(1).texts[0].new
plt.figure(1).texts[0].set_position([0.474888, 0.048140])
plt.figure(1).texts[0].set_text("Iterations")
#% end: automatic generated code from pylustrator
plt.show()