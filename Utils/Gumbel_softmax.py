import torch
import torch.nn.functional as F
import numpy as np


def inverse_gumbel_cdf(y, mu, beta):
    return mu - beta * np.log(-np.log(y))

def gumbel_softmax_sampling(h, mu=0, beta=1, tau=0.1):
    shape_h = h.shape
    p = F.softmax(h, dim=1)
    y = torch.rand(shape_h) + 1e-25  # ensure all y is positive
    g = inverse_gumbel_cdf(y, mu, beta)
    x = torch.log(p) + g # samples follow Gumbel distribution

    # using softmax to generate ont_hot vector
    x = x / tau
    x = F.softmax(x, dim=1)  # x approximates a one_hot vector
    return x

N = 10 # 假设有N个独立的离散变量需要采样
K = 3 # 假设 每个离散变量有3个取值
h = torch.randn((N, K)) # 假设 h 是由一个神经网络输出的tensor

mu = 0
beta = 1
tau = 0.1

samples = gumbel_softmax_sampling(h, mu, beta, tau)
