import torch
import sys
from facial_recognition.lfw_test import test
from facial_recognition.clnet import resnet18_clnet, resnet34_clnet, resnet50_clnet, mobilenetv2_clnet
from facial_recognition.sphereface import sphereface4, sphereface10, sphereface20, sphereface36, sphereface64


if __name__ == '__main__':
    import logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    torch.manual_seed(1)
    model = sys.argv[1]
    lfw_dir = sys.argv[2]
    lfw_test_file = sys.argv[3]

    net = mobilenetv2_clnet(256, None, 160)
    net.train(False)
    net.load(sys.argv[1])
    test(net, lfw_dir, lfw_test_file)
