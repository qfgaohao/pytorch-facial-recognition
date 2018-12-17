import argparse
import os
import random
import warnings

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

from facial_recognition.clnet import resnet18_clnet, resnet34_clnet, resnet50_clnet
from facial_recognition.center_loss import CenterLoss
from facial_recognition import transforms

# Modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Add supports for schedulers and other nets such as MobileNetV2.

models = {'resnet18_clnet': resnet18_clnet, 'resnet34_clnet': resnet34_clnet, 'resnet50_clnet': resnet50_clnet}
model_names = list(models.keys())

parser = argparse.ArgumentParser(description='PyTorch Center Loss Net Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34_clnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-d', '--dim', default=128, type=int, metavar='N',
                    help='the dimension of embeddings (default: 128)')

parser.add_argument('--lambda', default=0.01, dest='lambda_', type=float, metavar='M',
                    help='lambda for center loss')
parser.add_argument('--alpha', default=0.5, type=float, metavar='M',
                    help='alpha for center loss')


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
parser.add_argument('--scheduler', default="multi-step", type=str,
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

    dataset = datasets.ImageFolder(
        args.data,
        transforms.train_transform)
    num_classes = len(dataset.classes)
    logging.info(f"Num of classes: {num_classes}")
    torch.manual_seed(1)
    num_samples = len(dataset)
    num_train = int(num_samples * 0.95)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_samples - num_train])
    logging.info(f"Train data size {len(train_dataset)}. Validation data size: {len(val_dataset)}.")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    model = models[args.arch](args.dim, num_classes)

    # define loss function (criterion) and optimizer
    cross_entropy_loss_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    center_loss_criterion = CenterLoss(args.dim, num_classes, args.alpha)

    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        model.load(args.resume)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
        center_loss_criterion.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        center_loss_criterion.cuda()
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            center_loss_criterion.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            center_loss_criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    last_epoch = args.start_epoch if args.start_epoch > 0 else -1
    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.epochs, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
       val_dataset,
       batch_size=args.batch_size, shuffle=False,
       num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        scheduler.step()
        train(train_loader, model, cross_entropy_loss_criterion, center_loss_criterion, optimizer, epoch)
        if epoch % args.validation_epochs == 0 or epoch == args.epochs - 1:
            val_accuracy, val_loss, val_cross_entropy_loss, val_center_loss = validate(val_loader, model, cross_entropy_loss_criterion, center_loss_criterion)
            logging.info(
                f"Epoch: {epoch}, "
                f"Accuracy: {val_accuracy:.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Validation Cross-entropy Loss {val_cross_entropy_loss:.4f}, "
                f"Validation Center Loss: {val_center_loss:.4f}"
            )
            model_path = os.path.join(args.model_dir, f"{args.arch}-Epoch-{epoch}-Loss-{val_loss}.pth")
            model.save(model_path)
            logging.info(f"Saved model {model_path}")


def train(train_loader, model, cross_entropy_loss_criterion, center_loss_criterion, optimizer, epoch):

    # switch to train mode
    model.train()

    train_loss = 0
    train_cross_entropy_loss = 0
    train_center_loss = 0
    train_accuracy = 0
    num = 0

    for i, (inputs, labels) in enumerate(train_loader):
        num += 1
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

        # compute output
        features, logits = model(inputs)
        cross_entropy_loss = cross_entropy_loss_criterion(logits, labels)
        center_loss = center_loss_criterion(features, labels)

        loss = cross_entropy_loss + args.lambda_ * center_loss

        train_loss += loss.data
        train_cross_entropy_loss += cross_entropy_loss.data
        train_center_loss += center_loss.data

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
                f"Accuracy: {train_accuracy / num:.4f}, "
                f"Training Loss: {train_loss / num:.4f}, "
                f"Training Cross-entropy Loss {train_cross_entropy_loss / num:.4f}, "
                f"Training Center Loss: {train_center_loss / num:.4f}"
            )
            train_loss = 0
            train_cross_entropy_loss = 0
            train_center_loss = 0
            train_accuracy = 0
            num = 0


def validate(val_loader, model, cross_entropy_loss_criterion, center_loss_criterion):

    model.train()

    with torch.no_grad():
        total_loss = 0
        total_cross_entropy_loss = 0
        total_center_loss = 0
        total_accuracy = 0
        num = 0
        for i, (inputs, labels) in enumerate(val_loader):
            # measure data loading time
            num += 1
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)

            # compute output
            features, logits = model(inputs)
            cross_entropy_loss = cross_entropy_loss_criterion(logits, labels)
            center_loss = center_loss_criterion(features, labels)

            loss = cross_entropy_loss + args.lambda_ * center_loss
            total_loss += loss.data
            total_cross_entropy_loss += cross_entropy_loss.data
            total_center_loss += center_loss.data

            _, pred = logits.topk(1, 1)
            pred = pred.t()
            correct = pred.eq(labels).sum().float()
            total_accuracy += correct.data / inputs.size(0)
        return total_accuracy / num, total_loss / num, total_cross_entropy_loss / num, total_center_loss / num


if __name__ == '__main__':
    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
