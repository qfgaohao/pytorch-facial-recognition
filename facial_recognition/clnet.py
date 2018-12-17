import torch
from torch import nn
from .resnet import resnet18, resnet34, resnet50
from torchvision import models


class CLNet(nn.Module):

    def __init__(self, base_net: nn.Module, dim: int, num_classes: int=None):
        super(CLNet, self).__init__()
        self.base_net = base_net

        if num_classes is not None:
            self.fc = nn.Linear(dim, num_classes)
        else:
            self.train(False)

    def forward(self, x: torch.Tensor):
        if self.training:
            features = self.base_net(x)
            logits = self.fc(features)
            return features, logits
        else:
            with torch.no_grad():
                features = self.base_net(x)
                embeddings = nn.functional.normalize(features, 2, 1)
                return embeddings

    def save(self, model_path: str):
        torch.save(self.state_dict(), model_path)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def train(self, mode=True):
        if mode and not hasattr(self, 'fc'):
            raise ValueError("To use the train mode, "
                             "you have to provide num_classes for the constructor of this class."
                             "Otherwise you can only use the test mode.")
        super().train(mode)


resnet18_clnet = lambda dim, num_classes: CLNet(resnet18(pretrained=False, num_classes=dim), dim, num_classes)
resnet34_clnet = lambda dim, num_classes: CLNet(resnet34(pretrained=False, num_classes=dim), dim, num_classes)
resnet50_clnet = lambda dim, num_classes: CLNet(resnet50(pretrained=False, num_classes=dim), dim, num_classes)