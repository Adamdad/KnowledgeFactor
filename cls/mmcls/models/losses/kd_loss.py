import re
from numpy import inf
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class Logits(nn.Module):
    '''
    Do Deep Nets Really Need to be Deep?
    http://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf
    '''

    def __init__(self):
        super(Logits, self).__init__()

    def forward(self, out_s, out_t):
        loss = F.mse_loss(out_s, out_t)

        return loss


@LOSSES.register_module()
class SoftTarget(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''

    def __init__(self, temperature):
        super(SoftTarget, self).__init__()
        self.T = temperature

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss


def InfoMin_loss(mu, log_var):
    shape = mu.shape
    if len(shape) == 2:
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    elif len(shape) == 1:
        # print(torch.mean(1 + log_var - mu ** 2 - log_var.exp()))
        return -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp())

def InfoMax_loss(x1, x2):
    x1 = x1 / (torch.norm(x1, p=2, dim=1, keepdim=True) + 1e-10)
    x2 = x2 / (torch.norm(x2, p=2, dim=1, keepdim=True) + 1e-10)
    bs = x1.size(0)
    s = torch.matmul(x1, x2.permute(1, 0))
    mask_joint = torch.eye(bs).cuda()
    mask_marginal = 1 - mask_joint

    Ej = (s * mask_joint).mean()
    Em = torch.exp(s * mask_marginal).mean()
    # decoupled comtrastive learning?!!!!
    # infomax_loss = - (Ej - torch.log(Em)) * self.alpha
    infomax_loss = - (Ej - torch.log(Em)) #/ Em
    return infomax_loss
