import argparse
import os
import random
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from facial_recognition.clnet import resnet18_clnet, resnet34_clnet, resnet50_clnet, mobilenetv2_clnet
from facial_recognition.center_loss import CenterLoss
from facial_recognition import transforms
from facial_recognition.sphereface import sphereface4, sphereface10, sphereface20, sphereface36, sphereface64, \
    mobilenet_sphereface
from facial_recognition.asoftmax_loss import ASoftmaxLoss
from facial_recognition.ccs_loss import CCSLoss
from facial_recognition.ccs_net import resnet18_ccs_net, resnet34_ccs_net, mobilenetv2_ccs_net


# Modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Add supports for schedulers and other nets such as MobileNetV2.


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
model_names = list(models.keys())

parser = argparse.ArgumentParser(description='PyTorch Facial Recognition Net Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34_clnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-d', '--dim', default=128, type=int, metavar='N',
                    help='the dimension of embeddings (default: 128)')

parser.add_argument('--input-size', '-i', type=str,
                    help='It can be a single int or two ints representing h and w. Height comes first.')

parser.add_argument('--cl-alpha', default=0.5, type=float, metavar='M',
                    help='alpha for center loss')
parser.add_argument('--cl-lambda', default=0.01, type=float, metavar='M',
                    help='lambda for center loss')
parser.add_argument('--ccs-lambda', default=0, type=float, metavar='M',
                    help='lambda for the cosine similarity loss')
parser.add_argument('--sf-min-lambda', default=10, type=float, metavar='M',
                    help='a paramter to control the lambda in sphereface.')
parser.add_argument('--sf-max-lambda', default=1000, type=float, metavar='M',
                    help='a paramter to control the lambda in sphereface.')


parser.add_argument('-m', default=4, type=int, metavar='N',
                    help='m in ASoftmax loss')
parser.add_argument('-o', '--optimizer', default='sgd', type=str,
                    help="The optimizer can be sgd, adam, r")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--scheduler', default=None, type=str,
                    help="Scheduler for SGD. It can one of multi-step or cosine")
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--model_dir', default='models/',
                    help='Directory for saving models')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')


def is_sphereface_net(arch):
    return arch.startswith('sphereface')


def is_clnet(arch):
    return arch.endswith('clnet')


def is_ccs_net(arch):
    return arch.endswith('ccs_net')


def main():
    global args
    args = parser.parse_args()
    logging.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    input_size = [int(v) for v in args.input_size.split(",")]
    if len(input_size) == 1:
        input_size = (input_size[0], input_size[0])
    assert len(input_size) == 2

    dataset = datasets.ImageFolder(
        args.data,
        transforms.train_transform(input_size))
    num_classes = len(dataset.classes)
    logging.info(f"Num of classes: {num_classes}")
    logging.info(f"Train data size {len(dataset)}")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None

    if is_sphereface_net(args.arch):
        model = models[args.arch](args.dim, input_size, num_classes)
    elif is_clnet(args.arch):
        model = models[args.arch](args.dim, num_classes, args.input_size)
    elif is_ccs_net(args.arch):
        model = models[args.arch](args.dim, num_classes)

    if args.resume:
        logging.info(f"Load pretrained model from {args.resume}.")
        model.load(args.resume)

    if is_clnet(args.arch):
        cross_entropy_loss_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        center_loss_criterion = CenterLoss(args.dim, num_classes, args.cl_alpha).cuda(args.gpu)
        criterion = lambda features, logits, labels: cross_entropy_loss_criterion(logits, labels) + \
                                           args.cl_lambda * center_loss_criterion(features, labels)
    elif is_sphereface_net(args.arch):
        criterion = ASoftmaxLoss(args.m)
    elif is_ccs_net(args.arch):
        cross_entropy_loss_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        ccs_loss_criterion = CCSLoss()
        criterion = lambda features, weight, logits, labels:\
            cross_entropy_loss_criterion(logits, labels) + \
            args.ccs_lambda * ccs_loss_criterion(features, weight, logits, labels)

    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        model.load(args.resume)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
        # center_loss_criterion.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        # center_loss_criterion.cuda()
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            # center_loss_criterion.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            # center_loss_criterion.cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        logging.fatal(f"Only SGD, Adam, and RMSProp are supported. However you may modify the code and add it.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    last_epoch = args.start_epoch if args.start_epoch > 0 else -1
    scheduler = None
    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.epochs, last_epoch=last_epoch)
    elif args.scheduler is not None:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if scheduler is not None:
            scheduler.step()
        loss = train(train_loader, model, criterion, optimizer, epoch)
        if epoch % args.validation_epochs == 0 or epoch == args.epochs - 1:
            model_path = os.path.join(args.model_dir, f"{args.arch}-Epoch-{epoch}-Loss-{loss:.4f}.pth")
            model.save(model_path)
            logging.info(f"Saved model {model_path}")


lambda_iter = 0


def train(train_loader, model, criterion, optimizer, epoch):
    global lambda_iter, args
    # switch to train mode
    model.train()

    train_loss = 0
    train_accuracy = 0
    num = 0
    all_loss = 0
    all_num = 0
    for i, (inputs, labels) in enumerate(train_loader):
        num += 1
        lambda_iter += 1
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

        # compute output
        if is_clnet(args.arch):
            features, logits = model(inputs)
        elif is_ccs_net(args.arch):
            features, weight, logits = model(inputs)
        else:
            features, logits = model(inputs)
        lambda_ = 0.0
        if is_sphereface_net(args.arch):
            factor = (args.sf_max_lambda/args.sf_min_lambda - 1) / 2000
            lambda_ = max(args.sf_max_lambda / (factor * lambda_iter + 1), args.sf_min_lambda)
            loss = criterion(features, logits, labels, lambda_)
        elif is_clnet(args.arch):
            loss = criterion(features, logits, labels)
        elif is_ccs_net(args.arch):
            loss = criterion(features, weight, logits, labels)
        else:
            raise ValueError(f"The net {args.arch} is not supported.")
        train_loss += loss.data
        if math.isnan(train_loss):
            logging.warning("Nan Loss.")
            sys.exit(0)
        _, pred = logits.topk(1, 1)
        pred = pred.t()
        correct = pred.eq(labels).sum().float()
        train_accuracy += correct.data / inputs.size(0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            logging.info(
                f"Epoch: {epoch}, "
                f"Iter: {i}, "
                f"Lambda: {lambda_:.2f},"
                f"Accuracy: {train_accuracy / num:.4f}, "
                f"Training Loss: {train_loss / num:.4f}"
            )
            all_loss += train_loss
            all_num += num
            train_loss = 0
            train_accuracy = 0
            num = 0
    mean_loss = all_loss / (all_num + 1**-10)
    logging.info(f"Epoch: {epoch}, Loss: {mean_loss}.")
    return mean_loss


def validate(val_loader, model):

    model.train()

    with torch.no_grad():
        total_accuracy = 0
        num = 0
        for i, (inputs, labels) in enumerate(val_loader):
            # measure data loading time
            num += 1
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)
            if is_clnet(args.arch):
                _, logits, _ = model(inputs)
            else:
                _, logits = model(inputs)

            _, pred = logits.topk(1, 1)
            pred = pred.t()
            correct = pred.eq(labels).sum().float()
            total_accuracy += correct.data / inputs.size(0)
        return total_accuracy / num


if __name__ == '__main__':
    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
