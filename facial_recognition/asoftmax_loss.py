import torch
from torch import nn
import math

# def cos(logits, m):
#     computed = {}
#     return _cos(logits, m, computed)
#
#
# def _cos(logits, m, computed):
#     if m in computed:
#         return computed[m]
#     elif m == 1:
#         return logits
#     elif m == 0:
#         return 1
#     else:
#         a = _cos(logits, m - 1, computed)
#         b = _cos(logits, m - 2, computed)
#         c = 2 * a * logits - b
#         # if m not in computed:
#         #     computed[m] = c
#         return c

debug_iter = 0


def cos(logits, m):
    global debug_iter
    if m == 0:
        return 1
    elif m == 1:
        return logits
    elif m == 2:
        return torch.min(2 * logits**2 - 1, logits)
    elif m == 3:
        return torch.min(-3 * logits + 4 * logits**3, logits)
    elif m == 4:
        v = 1 - 8 * logits**2 + 8 * logits**4
        # print(torch.cat([logits, v], dim=1))
        mask0 = ((logits <= 1) & (logits > math.sqrt(2) / 2)).to(torch.float)
        mask1 = ((logits <= math.sqrt(2) / 2) & (logits > 0)).to(torch.float)
        mask2 = ((logits <= 0) & (logits > -math.sqrt(2) / 2)).to(torch.float)
        mask3 = ((logits <= -math.sqrt(2) / 2) & (logits >= -1)).to(torch.float)
        debug_iter += 1
        if debug_iter % 50 == 0:
            print("theta/m distribution: ",
                  mask0.sum().item(), mask1.sum().item(), mask2.sum().item(), mask3.sum().item())
        return (
            mask0 * v +
            mask1 * (-v - 2) +
            mask2 * (v - 4) +
            mask3 * (-v - 6)
        )
    else:
        raise ValueError(f"m {m} is not supported.")


class ASoftmaxLoss(nn.Module):

    def __init__(self, m: int, reduce=True):
        super(ASoftmaxLoss, self).__init__()
        self.m = m
        self.reduce = reduce

    def forward(self, features, logits, labels, annealing_lambda=0):
        global debug_iter
        norm = (features * features).sum(dim=1, keepdim=True).sqrt()

        sub_logits = logits.gather(1, labels.view(-1, 1))
        cos_logits = sub_logits / norm
        # if debug_iter % 50 == 0:
        #     print('features', features.data)
        #     print('norm', norm)
        #     print('sub_logits', sub_logits)
        #     print('cos_logits', cos_logits)
        #     print('featres-mean', features.mean(dim=1))
        #     print('featres-std', features.std(dim=1))
        m_cos_logits = cos(cos_logits, self.m)
        # print('m_cos_logits', m_cos_logits.min().data, m_cos_logits.max().data)
        # print('cos_logits', cos_logits.min().data, cos_logits.max().data)
        mlogits = (annealing_lambda * sub_logits + m_cos_logits * norm) / (1 + annealing_lambda)
        logits = torch.cat([logits, mlogits], 1)
        max_logit, _ = torch.max(logits, 1, keepdim=True)
        max_logit = max_logit.clamp(min=0)
        mlogits = mlogits - max_logit
        logits = logits - max_logit
        # print('--sub_logts', sub_logits.min().data, sub_logits.max().data)
        sub_logits = sub_logits - max_logit
        # print('sub_logts', sub_logits.min().data, sub_logits.max().data)
        loss = torch.exp(mlogits) / (torch.sum(torch.exp(logits), 1, keepdim=True) - torch.exp(sub_logits))
        # print('loss', loss.min().data, loss.max().data)
        loss = -torch.log(loss)
        if self.reduce:
            loss = torch.mean(loss)
        return loss

