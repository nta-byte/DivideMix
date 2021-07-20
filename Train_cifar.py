from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
# from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_cifar as dataloader
from utils import create_model, SemiLoss, NegEntropy, warmup, eval_train, train, test

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.85, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default=True, type=bool)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='/mnt/data1/ntanh/data/cifar-10-batches-py', type=str,
                    help='path to dataset')
parser.add_argument('--checkpoints_path', default='cp.pkl', type=str, help='path to dataset')
parser.add_argument('--model', default='mod_resnet18', type=str, help='model to choose: resnet18 or mod_resnet18')

parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--renew', default=False, type=bool)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def main():
    save_dir = f'./checkpoint/{args.model}_{args.dataset}_NoiseRate_{args.r}_NoiseMode_{args.noise_mode}_NumEpoch_{args.num_epochs}'
    if args.renew:
        if os.path.exists(save_dir):
            import shutil
            shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    args.checkpoints_path = os.path.join(save_dir, args.checkpoints_path)
    args.resume_path = args.checkpoints_path
    # stats_log = open(os.path.join(save_dir, '%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt'), 'w')
    acc_file = os.path.join(save_dir, '%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt')
    if not os.path.exists(acc_file):
        test_log = open(acc_file, 'w')
    else:
        test_log = open(acc_file, 'a')

    stat_file = os.path.join(save_dir, '%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt')
    if not os.path.exists(stat_file):
        stats_log = open(stat_file, 'w')
    else:
        stats_log = open(stat_file, 'a')
    warm_up = 0
    if args.dataset == 'cifar10':
        warm_up = 10
    elif args.dataset == 'cifar100':
        warm_up = 30

    loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                         num_workers=5,
                                         root_dir=args.data_path, log=stats_log,
                                         noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode),
                                         drop_last=False, pinmem=True)

    print('| Building net')
    net1 = create_model(args)
    net2 = create_model(args)
    cudnn.benchmark = True

    criterion = SemiLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    if args.noise_mode == 'asym':
        conf_penalty = NegEntropy()
    else:
        conf_penalty = None
    max_acc = 0
    lr = args.lr
    if args.resume:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'".format(args.resume_path))
            checkpoint = torch.load(args.resume_path)
            args.start_epoch = checkpoint['epoch']
            net1_dict = net1.state_dict()
            net2_dict = net2.state_dict()

            pretrained_dict1 = {k: v for k, v in checkpoint['state_dict1'].items() if
                                k in net1_dict and v.size() == net1_dict[k].size()}

            pretrained_dict2 = {k: v for k, v in checkpoint['state_dict2'].items() if
                                k in net2_dict and v.size() == net2_dict[k].size()}

            # print(len(pretrained_dict.keys()), len(model_dict.keys()))
            net1_dict.update(pretrained_dict1)
            net2_dict.update(pretrained_dict2)

            net1.load_state_dict(net1_dict)
            net2.load_state_dict(net2_dict)
            optimizer1.load_state_dict(checkpoint['optimizer1'])
            optimizer2.load_state_dict(checkpoint['optimizer2'])

            max_acc = checkpoint['max_acc']
            lr = checkpoint['lr']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_path))
            lr = args.lr

    all_loss = [[], []]  # save the history of losses from two networks

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        # for epoch in range(args.num_epochs + 1):

        if epoch >= 150:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr
        test_loader = loader.run('test')
        eval_loader = loader.run('eval_train')

        if epoch < warm_up:
            warmup_trainloader = loader.run('warmup')
            print('Warmup Net1')
            warmup(args, epoch, net1, optimizer1, warmup_trainloader, CEloss, conf_penalty)
            print('\nWarmup Net2')
            warmup(args, epoch, net2, optimizer2, warmup_trainloader, CEloss, conf_penalty)

        else:
            prob1, all_loss[0] = eval_train(args, net1, all_loss[0], CE, eval_loader)
            prob2, all_loss[1] = eval_train(args, net2, all_loss[1], CE, eval_loader)

            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)  # co-divide
            train(args, epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader, criterion,
                  warm_up)  # train net1

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
            train(args, epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader, criterion,
                  warm_up)  # train net2

        val_acc = test(epoch, net1, net2, test_log, test_loader)
        if val_acc > max_acc:
            net1_path = os.path.join(save_dir, f"net1_resnet18-best.pth")
            net2_path = os.path.join(save_dir, f"net2_resnet18-best.pth")

            torch.save(net1.state_dict(), net1_path)
            torch.save(net2.state_dict(), net2_path)
            max_acc = val_acc
            print(f"Saved net1 {net1_path}")
            print(f"Saved net2 {net2_path}")

        torch.save({'epoch': epoch + 1,
                    'state_dict1': net1.state_dict(),
                    'state_dict2': net2.state_dict(),
                    'optimizer1': optimizer1.state_dict(),
                    'optimizer2': optimizer2.state_dict(),
                    'lr': lr,
                    'max_acc': max_acc
                    },

                   args.checkpoints_path)


if __name__ == '__main__':
    main()
