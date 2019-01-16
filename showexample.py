import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import os
import re


def show_result(datafolder, cname, num = 100, folder = 'imgs/'):

    data = np.load(datafolder + cname + '.npy')
    ids = list(range(len(data)))
    random.shuffle(ids)
    ids = ids[:100]
    data = np.array([data[x] for x in ids])

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(data[k].reshape(28, 28))

    label = cname
    fig.text(0.5, 0.04, label, ha='center')
    print('saving ' + folder + cname + '.png')
    plt.savefig(folder + cname + '.png')

datafolder = 'data/quickdraw/'
files = [x[:-4] for x in os.listdir(datafolder)]
for file in files:
    show_result(datafolder, file)