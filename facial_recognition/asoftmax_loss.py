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
        v = 2 * logits**2 - 1
        mask0 = logits > 0
        mask1 = logits <= 0
        debug_iter += 1
        if debug_iter % 50 == 0:
            print("theta/m distribution: ",
                  mask0.sum().item(), mask1.sum().item())
        return (
            mask0 * v +
            mask1 * (-v - 2)
        )
    elif m == 3:
        v = -3 * logits + 4 * logits**3
        pivot = math.cos(math.pi/3)
        mask0 = ((logits <=1) & (logits > pivot)).to(torch.float)
        mask1 = ((logits <= pivot) & logits > -pivot).to(torch.float)
        mask2 = ((logits <= -pivot) & logits >= -1).to(torch.float)
        if debug_iter % 50 == 0:
            print("theta/m distribution: ",
                  mask0.sum().item(), mask1.sum().item(), mask2.sum().item())
        return (
            mask0 * v +
            mask1 * (-v - 2) +
            mask2 * (v - 4)
        )
    elif m == 4:
        v = 1 - 8 * logits**2 + 8 * logits**4
        pivot = math.cos(math.pi/4)
        mask0 = ((logits <= 1) & (logits > pivot)).to(torch.float)
        mask1 = ((logits <= pivot) & (logits > 0)).to(torch.float)
        mask2 = ((logits <= 0) & (logits > - pivot)).to(torch.float)
        mask3 = ((logits <= -pivot) & (logits >= -1)).to(torch.float)
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

    def __init__(self, m: int, ):
        super(ASoftmaxLoss, self).__init__()
        self.m = m

    def forward(self, features, logits, labels, annealing_lambda=0):
        global debug_iter

        features_norm = (features ** 2).sum(dim=1, keepdim=True).sqrt()
        x_cos_theta = logits.gather(1, labels.view(-1, 1))
        cos_theta = x_cos_theta / features_norm
        cos_theta = cos_theta.clamp(-1, 1)
        x_cos_theta = cos_theta * features_norm
        cos_m_theta = cos(cos_theta, self.m)
        x_cos_m_theta = features_norm * cos_m_theta
        f_y = (annealing_lambda * x_cos_theta + x_cos_m_theta) / (1 + annealing_lambda)
        logits = logits.scatter(1, labels.view(-1, 1), f_y)
        return torch.nn.functional.cross_entropy(logits, labels)
