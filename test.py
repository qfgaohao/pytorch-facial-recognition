import torch
import sys
import argparse

from facial_recognition.lfw_test import test
from facial_recognition.clnet import resnet18_clnet, resnet34_clnet, resnet50_clnet, mobilenetv2_clnet
from facial_recognition.sphereface import sphereface4, sphereface10, sphereface20, \
    sphereface36, sphereface64, mobilenet_sphereface
from facial_recognition.ccs_net import resnet18_ccs_net, resnet34_ccs_net, mobilenetv2_ccs_net


models = {
    'resnet18_clnet': resnet18_clnet,
    'resnet34_clnet': resnet34_clnet,
    'resnet50_clnet': resnet50_clnet,
    'mobilenetv2_clnet': mobilenetv2_clnet,
    'sphereface4': sphereface4,
    'sphereface10': sphereface10,
    'sphereface20': sphereface20,
    'sphereface36': sphereface36,
    'sphereface64': sphereface64,
    'sphereface_mb2': mobilenet_sphereface,
    'resnet18_ccs_net': resnet18_ccs_net,
    'resnet34_ccs_net': resnet34_ccs_net,
    'mobilenetv2_ccs_net': mobilenetv2_ccs_net
}


def is_sphereface_net(arch):
    return arch.startswith('sphereface')


def is_clnet(arch):
    return arch.endswith('clnet')


def is_ccs_net(arch):
    return arch.endswith('ccs_net')


model_names = list(models.keys())

parser = argparse.ArgumentParser(description='PyTorch Facial Recognition Net Training')
parser.add_argument('--lfw-dir', '-l', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--lfw-file', '-p', metavar='FILE', help='The LFW file containing pairs to test.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34_clnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--trained-model', '-t', help="The trained model")
parser.add_argument('-d', '--dim', default=128, type=int, metavar='N',
                    help='the dimension of embeddings (default: 128)')
parser.add_argument('--input-size', '-i', type=str,
                    help='It can be a single int or two ints representing w and h')

parser.add_argument('--metric', '-m', metavar='Metric', default='l2',
                    help='It can be l2 for L2-distance or cosine for cosine distance.')
parser.add_argument('--flip', '-f', default=False, action='store_true',
                    help='if true, also add horizontally flipped image to test.')

if __name__ == '__main__':
    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    args = parser.parse_args(sys.argv[1:])
    torch.manual_seed(1)

    input_size = [int(v) for v in args.input_size.split(",")]
    if len(input_size) == 1:
        input_size = (input_size[0], input_size[0])
    assert len(input_size) == 2

    if is_sphereface_net(args.arch):
        net = models[args.arch](args.dim)
    elif is_clnet(args.arch):
        net = models[args.arch](args.dim)
    elif is_ccs_net(args.arch):
        net = models[args.arch](args.dim)
    net.train(False)
    net.load(args.trained_model)
    test(net, args.lfw_dir, args.lfw_file, input_size, args.metric, args.flip)
