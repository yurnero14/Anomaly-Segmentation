import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OhemCELoss(nn.Module):

    def __init__(self, thresh, weight=None):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        if weight is not None:
          self.criteria = nn.CrossEntropyLoss(reduction='none',weight=weight)
        else:
          self.criteria = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, logits, labels):
        n_min = labels.numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


