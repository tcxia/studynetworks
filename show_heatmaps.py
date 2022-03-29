from numpy import mat
import torch
from IPython import display
import matplotlib.pyplot as plt



def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    display.set_matplotlib_formats('svg')
    nrows, ncols = matrices.shape[0], matrices.shape[1]
    print(nrows, ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            print(i, j)
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == nrows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)