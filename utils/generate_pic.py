import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
import torch.utils.data as Data



def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 1:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
    return y


def generate_png(pred, target,epoch):
    pred_test = np.array(pred).flatten()


    gt = target.flatten()
    x = np.ravel(pred_test)
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    y_re = np.reshape(y_list, (target.shape[1], target.shape[2], 3))
    gt_re = np.reshape(y_gt, (target.shape[1], target.shape[2], 3))
    target=target.reshape(512,512)
    path = '/opt/data/private/FSVLM/imgs/'
    classification_map(y_re, target, 300,
                       path +  str(epoch)+ '_' + '.png')
    classification_map(gt_re, target, 300,
                       path +  str(epoch) + '_gt.png')
    print('------Get classification maps successful-------')
