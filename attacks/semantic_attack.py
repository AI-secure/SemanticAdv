import torch
import torch.nn as nn
from torch import optim

from .utils import TVLoss


'''
Transform the G(x) to G_dec(G_enc(x)) before use this function
'''

class FP_CW(nn.Module):
    def __init__(self, lr = 0.05, max_iteration = 200, early_stop = None, initial_alpha=0.8, max_alpha=1.0, min_alpha=0.6):
        super(FP_CW, self).__init__()
        self.lr = lr
        self.max_iteration = max_iteration
        self.early_stop = early_stop
        self.initial_alpha = initial_alpha
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha

    def forward(self, G_dec, emb1, emb2, model, loss_func, target_label, targeted = True):
        '''
        :param G_dec:
        :param emb1: original feature map
        :param emb2: new feature map
        :param model: model to attack
        :param loss_func:
        :param gt_label:
        :return:
        '''
        assert emb1.shape == emb2.shape
        k = torch.zeros_like(emb1) + self.initial_alpha
        k_max = torch.zeros_like(emb1) + self.max_alpha
        k_min = torch.zeros_like(emb1) + self.min_alpha
        k = k.cuda()
        k.requires_grad = True
        optimizer = optim.Adam([k], lr = self.lr)
        for z in range(self.max_iteration + 1):
            out = G_dec((1-torch.min(torch.max(k,k_min),k_max)) * emb1 + torch.min(torch.max(k,k_min),k_max) * emb2)
            adv_loss = loss_func(model(out), target_label)
            if targeted == False:
                adv_loss = -adv_loss
            if self.early_stop is not None and adv_loss < self.early_stop:
                break

            optimizer.zero_grad()
            adv_loss.backward()
            optimizer.step()

        return out, adv_loss
        # return out

class FP_CW_TV(nn.Module):
    def __init__(self, lr = 0.05, max_iteration = 200, tv_lambda = 0.01, early_stop = None, initial_alpha=0.8, max_alpha=1.0, min_alpha=0.6):
        super(FP_CW_TV, self).__init__()
        self.lr = lr
        self.max_iteration = max_iteration
        self.tv_lambda = tv_lambda
        self.early_stop = early_stop
        self.initial_alpha = initial_alpha
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.criterionTV = TVLoss(L=2)

    def forward(self, G_dec, emb1, emb2, model, loss_func, target_label, targeted = True):
        '''
        :param G_dec:
        :param emb1: original feature map
        :param emb2: new feature map
        :param model: model to attack
        :param loss_func:
        :param gt_label:
        :return:
        '''
        assert emb1.shape == emb2.shape
        k = torch.zeros_like(emb1) + self.initial_alpha
        k_max = torch.zeros_like(emb1) + self.max_alpha
        k_min = torch.zeros_like(emb1) + self.min_alpha
        k = k.cuda()
        k.requires_grad = True
        optimizer = optim.Adam([k], lr = self.lr)
        for z in range(self.max_iteration + 1):
            out = G_dec((1-torch.min(torch.max(k,k_min),k_max)) * emb1 + torch.min(torch.max(k,k_min),k_max) * emb2)
            adv_loss = loss_func(model(out), target_label)
            if targeted == False:
                adv_loss = -adv_loss
            tv_loss = self.criterionTV(k)
            loss = adv_loss + tv_loss * self.tv_lambda
            if self.early_stop is not None and adv_loss < self.early_stop:
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return out, adv_loss, tv_loss
        # return out

#TODO FP_PGD
