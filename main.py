#!/usr/bin/env python3

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from dcgan import DCGAN 
from sagan import SAGAN 

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default="data", help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--gan', default="dc", help='dc | sa: use either Deep Convolutional GAN or Self Attention GAN')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='output', help='folder to output images')
parser.add_argument('--checkpoint', default='checkpoint', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

cudnn.benchmark = True
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


def load_dataset():
    transform = transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])
    train = dset.CIFAR10(root=opt.dataroot, download=True, train=True, transform=transform)
    return train


def to_dataloader(trainset):
    train_dataloader = torch.utils.data.DataLoader(trainset, 
            batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    return train_dataloader


def train(gan, train_dataloader):
    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    netG = gan.netG
    netD = gan.netD
    for epoch in range(opt.niter):
        for i, data in enumerate(train_dataloader, 0):
            print('[%d/%d][%d/%d] '
                  % (epoch, opt.niter, i, len(train_dataloader),), end="")

            gan.train_on_batch(data, device)
    
            if i % 100 == 0:
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
    
        # save checkpoint
        torch.save(netG.state_dict(), '%s/netG/netG_epoch_%d.pth' % (opt.checkpoint, epoch))
        torch.save(netD.state_dict(), '%s/netD/netD_epoch_%d.pth' % (opt.checkpoint, epoch))

###### MAIN

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Prepare dataset
train_dataloader = to_dataloader(load_dataset())

if opt.gan == "sa":
    print("Loading Self Attention GAN")
    gan = SAGAN(nc=nc, nz=nz, ngf=ngf, ndf=ndf, ngpu=ngpu)
else:
    print("Loading Deep Convolutional GAN")
    gan = DCGAN(nc=nc, nz=nz, ngf=ngf, ndf=ndf, ngpu=ngpu)

# Init Generator
if opt.netG != '':
    gan.netG.load_state_dict(torch.load(opt.netG))
print(gan.netG)

# Init Discriminator
if opt.netD != '':
    gan.netD.load_state_dict(torch.load(opt.netD))
print(gan.netD)

train(gan, train_dataloader)

