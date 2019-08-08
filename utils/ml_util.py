import numpy as np
import torch
from torch import nn


def init_unk(tensor_, vocab_size):
    bound = np.sqrt(6.9) / np.sqrt(vocab_size)
    return torch.from_numpy(np.random.uniform(-bound, bound, tensor_.shape))


def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name and w.dim() > 1:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def count_params(model):
    """
    打印该模型中待训练的参数个数
    :param model:
    :return:
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adjust_lr(optimizer, cur_epoch, init_lr, decay_rate, min_lr):
    """
    调整优化器的学习率, 该函数在每个epoch循环时调用:
        ```
            for epoch in range(epochs):
                adjust_lr(sgd_optimizer, epoch)
                train(...)
                validate(...)
        ```
    :param optimizer: 优化器对象
    :param cur_epoch: 当前迭代epoch
    :param init_lr: 初始化的学习率
    :param decay_rate: 每步迭代衰减率
    :param min_lr: 最小学习率，低于该值时截断
    :return: 设置优化器对象中参数的学习率
    """
    cur_lr = init_lr * (decay_rate ** cur_epoch)
    if cur_lr < min_lr:
        cur_lr = min_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = cur_lr

    return cur_lr
