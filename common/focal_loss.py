import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, gamma=2., reduction='sum'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        
        # 'none' | 'mean' | 'sum
        self.reduction = reduction
        
        
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target, 
            weight=self.weight,
            reduction = self.reduction
        )