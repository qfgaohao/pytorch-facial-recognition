import torch
from torch import nn
from .resnet import resnet18, resnet34, resnet50
from .mobilenet_v2 import MobileNetV2


class CCSNet(nn.Module):

    def __init__(self, base_net: nn.Module, dim: int, num_classes: int=None):
        super(CCSNet, self).__init__()
        self.base_net = base_net

        if num_classes is not None:
            self.fc = nn.Linear(dim, num_classes, bias=False)
        else:
            self.train(False)

    def forward(self, x: torch.Tensor):
        if self.training:
            features = self.base_net(x)
            logits = self.fc(features)
            return features, self.fc.weight, logits
        else:
            with torch.no_grad():
                features = self.base_net(x)
                embeddings = nn.functional.normalize(features, 2, 1)
                return embeddings

    def save(self, model_path: str):
        torch.save(self.state_dict(), model_path)

    def load(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        if not hasattr(self, 'fc2'):
            state_dict = {k: v for k, v in state_dict.items() if k not in set(["fc.weight"])}
        self.load_state_dict(state_dict)

    def train(self, mode=True):
        if mode and not hasattr(self, 'fc'):
            raise ValueError("To use the train mode, "
                             "you have to provide num_classes for the constructor of this class."
                             "Otherwise you can only use the test mode.")
        super().train(mode)


resnet18_ccs_net = lambda dim, num_classes=None: CCSNet(resnet18(pretrained=False, num_classes=dim), dim, num_classes)
resnet34_ccs_net = lambda dim, num_classes=None: CCSNet(resnet34(pretrained=False, num_classes=dim), dim, num_classes)
mobilenetv2_ccs_net = lambda dim, num_classes=None, input_size=160: CCSNet(MobileNetV2(dim, input_size=input_size, onnx_compatible=True),
                                                   dim, num_classes)