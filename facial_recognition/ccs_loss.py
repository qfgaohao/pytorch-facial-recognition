import torch
from torch import nn


class CCSLoss(nn.Module):

    def __init__(self):
        super(CCSLoss, self).__init__()

    def forward(self, features, weight, logits, labels):
        sub_logits = logits.gather(1, labels.view(-1, 1))
        weight_norm = (weight**2).sum(dim=1, keepdim=True).sqrt()
        weight_norm = weight_norm[labels]
        features_norm = (features ** 2).sum(dim=1, keepdim=True).sqrt()
        normalized_logits = sub_logits / weight_norm / features_norm
        loss = 1 - normalized_logits.mean()
        return loss