import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1, size_average = True, L = 1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.size_average = size_average
        self.L = L

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        if self.L == 1:
            h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
            w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        else:
            h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :],2).sum()
            w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1],2).sum()
        if not self.size_average:
            return self.TVLoss_weight*(h_tv + w_tv)
        else:
            count_h = self._tensor_size(x[:,:,1:,:])
            count_w = self._tensor_size(x[:,:,:,1:])
            return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
