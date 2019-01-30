import torch
from torch import nn
from .mobilenet_v2 import MobileNetV2


class Block(nn.Module):
    def __init__(self, num_residual_layers, in_channels, out_channels,
                 kernel_size=3, stride=2, padding=1, remove_last_relu=False):
        super(Block, self).__init__()
        if remove_last_relu and num_residual_layers == 0:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.PReLU()
            )
        layers = []
        for i in range(num_residual_layers):
            if remove_last_relu and i + 1 == num_residual_layers:
                layer = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.PReLU(),
                    nn.BatchNorm2d(out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.PReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.PReLU()
                )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.conv(x)
        for layer in self.layers:
            residual = layer(x)
            x = x + residual
        return x


class AngularLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AngularLinear, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        logits = self.fc(x)
        weight_norm = (self.fc.weight ** 2).sum(dim=1, keepdim=True).sqrt()
        logits = logits / weight_norm.t()
        return logits


class SpereFaceNet(nn.Module):

    def __init__(self, input_size, dim: int, num_residual_layers_per_block, out_channels_per_block):
        super(SpereFaceNet, self).__init__()
        blocks = []
        in_channels = 3
        for i, (num, out_channels) in enumerate(zip(num_residual_layers_per_block, out_channels_per_block)):
            remove_last_relu = (i + 1 == len(num_residual_layers_per_block))
            block = Block(num, in_channels, out_channels, remove_last_relu=remove_last_relu)
            in_channels = out_channels
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        assert len(input_size) == 2
        assert input_size[0] % 16 == 0
        assert input_size[1] % 16 == 0
        feature_map_size = (int(input_size[0]/16), int(input_size[1]/16))
        self.fc = nn.Linear(feature_map_size[0] * feature_map_size[1] * out_channels_per_block[-1], dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        return features


class SphereFace(nn.Module):
    def __init__(self, base_net, dim: int, num_classes: int=None):
        super(SphereFace, self).__init__()
        self.base_net = base_net
        if num_classes is not None:
            self.fc = AngularLinear(dim, num_classes)

    def forward(self, x):
        x = self.base_net(x)
        if self.training:
            # normalize weight per class
            logits = self.fc(x)
            return x, logits
        else:
            return x

    def save(self, model_path: str):
        torch.save(self.state_dict(), model_path)

    def load(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        if not hasattr(self, 'fc'):
            state_dict = {k: v for k, v in state_dict.items() if k not in set(["fc.fc.weight"])}
        self.load_state_dict(state_dict)


def mobilenet_sphereface(dim=512, input_size=160, num_classes: int=None):
    base_net = MobileNetV2(n_class=dim, input_size=input_size, width_mult=1.,
                 use_batch_norm=True, onnx_compatible=True)
    net = SphereFace(base_net, dim, num_classes)
    return net


def sphereface4(dim=512, input_size=(112, 96), num_classes: int=None):
    base_net = SpereFaceNet(input_size, dim, [0, 0, 0, 0], [64, 128, 256, 512])
    net = SphereFace(base_net, dim, num_classes)
    return net


def sphereface10(dim=512, input_size=(112, 96), num_classes: int=None):
    base_net = SpereFaceNet(input_size, dim, [0, 1, 2, 0], [64, 128, 256, 512])
    net = SphereFace(base_net, dim, num_classes)
    return net


def sphereface20(dim=512, input_size=(112, 96), num_classes: int=None):
    base_net = SpereFaceNet(input_size, dim, [1, 2, 4, 1], [64, 128, 256, 512])
    net = SphereFace(base_net, dim, num_classes)
    return net


def sphereface36(dim=512, input_size=(112, 96), num_classes: int=None):
    base_net = SpereFaceNet(input_size, dim, [1, 4, 8, 2], [64, 128, 256, 512])
    net = SphereFace(base_net, dim, num_classes)
    return net


def sphereface64(dim=512, input_size=(112, 96), num_classes: int=None):
    base_net = SpereFaceNet(input_size, dim, [3, 8, 16, 3], [64, 128, 256, 512])
    net = SphereFace(base_net, dim, num_classes)
    return net
