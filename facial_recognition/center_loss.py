import torch
from torch import nn


class CenterLoss(nn.Module):

    def __init__(self, dim: int, num_classes: int, alpha: float):
        super(CenterLoss, self).__init__()
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, dim, dtype=torch.float), requires_grad=False)

    def forward(self, features, labels):
        active_centers = self.centers[labels]
        diff = active_centers - features
        loss = torch.mean(diff ** 2)
        with torch.no_grad():
            # update centers
            label_counts = torch.bincount(labels)
            label_counts = label_counts[labels]
            weights = 1 / label_counts.to(torch.float)
            diff = diff * weights.unsqueeze(1) * self.alpha
            self.centers.index_add_(0, labels, -diff)

        return loss

