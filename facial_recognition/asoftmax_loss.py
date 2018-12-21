import torch
from torch import nn


def cos(logits, m):
    computed = {}
    return _cos(logits, m, computed)


def _cos(logits, m, computed):
    if m in computed:
        return computed[m]
    elif m == 1:
        return logits
    elif m == 0:
        return 1
    else:
        a = _cos(logits, m - 1, computed)
        b = _cos(logits, m - 2, computed)
        c = 2 * a * logits - b
        if m not in computed:
            computed[m] = c
        return c


class ASoftmaxLoss(nn.Module):

    def __init__(self, m: int, reduce=True):
        super(ASoftmaxLoss, self).__init__()
        self.m = m
        self.reduce = reduce

    def forward(self, logits, labels, annealing_lambda=0):
        sub_logits = logits.gather(1, labels.view(-1, 1))
        #print('logits', logits.size())
        mlogits = cos(sub_logits, self.m)
        mlogits = (annealing_lambda * sub_logits + mlogits) / (1 + annealing_lambda)
        right_loss = torch.nn.functional.cross_entropy(logits, labels)
        #mlogits = sub_logits
        #print(mlogits)
        logits = torch.cat([logits, mlogits], 1)
        #print('logits', logits.size())
        max_logit, _ = torch.max(logits, 1, keepdim=True)
        max_logit = max_logit.clamp(min=0)
        #print('max_logits', max_logit.size())
        mlogits = mlogits - max_logit
        logits = logits - max_logit
        sub_logits = sub_logits - max_logit
        loss = torch.exp(mlogits) / (torch.sum(torch.exp(logits), 1, keepdim=True) - torch.exp(sub_logits))
        loss = -torch.log(loss)
        if self.reduce:
            loss = torch.mean(loss)
        #print(right_loss, loss)
        return loss

