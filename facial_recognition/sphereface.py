import torch
from torch import nn


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


class SpereFace(nn.Module):

    def __init__(self, feature_map_size, dim: int, num_residual_layers_per_block, out_channels_per_block, num_classes: int=None):
        super(SpereFace, self).__init__()
        blocks = []
        in_channels = 3
        for i, (num, out_channels) in enumerate(zip(num_residual_layers_per_block, out_channels_per_block)):
            remove_last_relu = (i + 1 == len(num_residual_layers_per_block))
            block = Block(num, in_channels, out_channels, remove_last_relu=remove_last_relu)
            in_channels = out_channels
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        if isinstance(feature_map_size, int):
            feature_map_size = (feature_map_size, feature_map_size)
        self.fc1 = nn.Linear(feature_map_size[0] * feature_map_size[1] * out_channels_per_block[-1], dim)
        if num_classes is not None:
            #self.fc2 = torch.nn.utils.weight_norm(nn.Linear(dim, num_classes, bias=False), 'weight')
            #print('weight', self.fc2.weight.size())
            self.fc2 = nn.Linear(dim, num_classes, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        features = self.fc1(x)
        if self.training:
            # normalize weight per class
            logits = self.fc2(features)
            with torch.no_grad():
                weight_norm = (self.fc2.weight**2).sum(dim=1, keepdim=True).sqrt()
            logits = logits / weight_norm.t()
            return features, logits
        else:
            return features

    def save(self, model_path: str):
        torch.save(self.state_dict(), model_path)

    def load(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        if not hasattr(self, 'fc2'):
            state_dict = {k: v for k, v in state_dict.items() if k not in set(["fc2.weight"])}
        self.load_state_dict(state_dict)


def sphereface4(feature_map_size, dim=512, num_classes: int=None):
    return SpereFace(feature_map_size, dim, [0, 0, 0, 0], [64, 128, 256, 512], num_classes=num_classes)


def sphereface10(feature_map_size, dim=512, num_classes: int=None):
    return SpereFace(feature_map_size, dim, [0, 1, 2, 0], [64, 128, 256, 512], num_classes=num_classes)


def sphereface20(feature_map_size, dim=512, num_classes: int=None):
    return SpereFace(feature_map_size, dim, [1, 2, 4, 1], [64, 128, 256, 512], num_classes=num_classes)


def sphereface36(feature_map_size, dim=512, num_classes: int=None):
    return SpereFace(feature_map_size, dim, [1, 4, 8, 2], [64, 128, 256, 512], num_classes=num_classes)


def sphereface64(feature_map_size, dim=512, num_classes: int=None):
    return SpereFace(feature_map_size, dim, [3, 8, 16, 3], [64, 128, 256, 512], num_classes=num_classes)
