import numpy as np
import matplotlib.pyplot as plt
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
cfg = {
    "colors": colors ,
    "alpha": 1.0,
    "fontsize": 35,
    "fontsize_legend": 20,
    "MarkerSize": 60,
    "linewidth": 5,
    "color_w": 0.5,
}
alpha = "alpha"
fontsize = "fontsize"
fontsize_legend = "fontsize_legend"
MarkerSize = "MarkerSize"
linewidth = "linewidth"
color_w = "color_w"

def plt_tick_1():
    # plt.ylim([-2.5, 2.5])
    # plt.xlim([-2.5, 2.5])
    plt.xticks([-5, -2.5, 0, 2.5, 5], ['$-5$', '', '$0$', '', '$5$'])
    plt.yticks([-5, -2.5, 0, 2.5, 5], ['$-5$', '', '$0$', '', '$5$'])
def plt_tick_2():
    # plt.ylim([-2.5, 2.5])
    plt.xticks([0, 2, 4, 6])
    plt.yticks([-5, -2.5, 0, 2.5, 5], ['$-5$', '', '$0$', '', '$5$'])


def plot_jianbian_line(
        X, Y, start_color=np.array([1.0, 0.0, 0.0]),
        end_color=np.array([0.0, 1.0, 0.0]),
        scale = 1/3,
        width_rate = 9/10,
):
    data_len = len(X)
    # plt.plot(data[0,:1000], data[1, :1000], '-', alpha=alpha)
    n = 500
    seg_len = data_len // n
    print('data_len:{}, n:{}, seg_len:{}'.format(data_len, n, seg_len))
    for i in range(n - 1):
        w = ((i) / n) ** (scale)
        now_color = start_color +  w * (end_color - start_color)
        print('i:{}, now_color:{}'.format(i, now_color))
        # plt.plot(data[0,i:i+3], data[1,i:i+3], '-', color=now_color, alpha=alpha)
        plt.plot(X[seg_len * i:seg_len * (i+1)], Y[seg_len * i:seg_len * (i+1)],
                 '-', color=now_color, alpha=alpha, linewidth= linewidth - w * linewidth * width_rate )

