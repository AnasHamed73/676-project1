#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
from scipy import linalg
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as dset
from torchvision.models import inception_v3
from sagan import SAGAN
import torchvision.transforms as transforms
import gc

ngpu =1 
nz = 100
ngf = 64
ndf = 64 
nc = 3
num_classes = 10

checkpoint = 84
prevNetG = "checkpoint/netG/netG_epoch_" + str(checkpoint) + ".pth"
fid_file = "./fid_scores"
fid_score_batch_size = 256


class Buffer(nn.Module):
    def __init__(self):
        super(Buffer, self).__init__()
						        
    def forward(self, x):
        return x


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def mean_cov(act):
    m = np.mean(act, axis=0)
    s = np.cov(act, rowvar=False)
    return m, s


def calc_fid(real, generated, model, device):
    real_resized = F.interpolate(real, size=(299, 299), mode='bilinear',
            align_corners=False)
    gen_resized = F.interpolate(generated, size=(299, 299), mode='bilinear',
            align_corners=False)
    real_resized = real_resized.to(device)
    gen_resized = gen_resized.to(device)
    real_activations = model(real_resized).cpu().detach().numpy()
    del real_resized
    gc.collect()
    gen_activations = model(gen_resized).cpu().detach().numpy()
    del gen_resized
    gc.collect()
    
    mu_real, sigma_real = mean_cov(real_activations)
    mu_gen, sigma_gen = mean_cov(gen_activations)
    score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print("%%%%%%%%%%%%%FID Score: ", score)
    return score


def load_inception(use_gpu=False):
    inception = inception_v3(pretrained=True)
    # Remove linear activation layer by replacing it with a buffer layer
    inception.fc = Buffer()
    inception.eval()
    if use_gpu:
        inception.cuda()
    return inception


def load_dataset():
    transform = transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])
    test = dset.CIFAR10(root="data", download=True, train=False, transform=transform)
    return test


def to_dataloader(testset):
    test_rand_sampler = torch.utils.data.RandomSampler(testset,
            replacement=True, num_samples=fid_score_batch_size)
    test_dataloader = torch.utils.data.DataLoader(testset,
            batch_size=fid_score_batch_size,
            sampler=test_rand_sampler, num_workers=1)
    return test_dataloader


def calc_fid_single_epoch(netG, inception, device, labels=None):
    # Prepare dataset
    testset = load_dataset()
    test_dataloader = to_dataloader(testset)
    with torch.no_grad():
        print("Calculating FID...")
        test_samples = next(iter(test_dataloader))[0]
        print("Loaded {0} real test samples".format(fid_score_batch_size))
        test_noise = torch.randn(fid_score_batch_size, nz, 1, 1, device=device)
        gen_samples = netG(test_noise) if labels is None else netG(test_noise, labels)
        print("Generated {0} fake samples".format(fid_score_batch_size))
        test_samples.to(device)
        gen_samples.to(device)
        return calc_fid(test_samples.detach(), gen_samples.detach(), inception, device)


def one_hot(labels, device):
    y_onehot = torch.FloatTensor(labels.size(0), num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, labels.unsqueeze(1), 1)
    return y_onehot.unsqueeze(2).unsqueeze(3).to(device, non_blocking=True)


def calc_fid_single_epoch_conditional(netG, inception, device):
    # Prepare dataset
    testset = load_dataset()
    test_dataloader = to_dataloader(testset)
    with torch.no_grad():
        print("Calculating FID...")
        real_sample = next(iter(test_dataloader))
        test_samples = real_sample[0]
        print("Loaded {0} real test samples".format(fid_score_batch_size))
        test_noise = torch.randn(fid_score_batch_size, nz, 1, 1, device=device)
        gen_samples = netG(test_noise, one_hot(real_sample[1], device))
        print("Generated {0} fake samples".format(fid_score_batch_size))
        test_samples.to(device)
        gen_samples.to(device)
        return calc_fid(test_samples.detach(), gen_samples.detach(), inception, device)



def calc_fid_all_epochs(netG, inception, device):
    # Prepare dataset
    testset = load_dataset()
    test_dataloader = to_dataloader(testset)
    with torch.no_grad():
        for i in range(83, 1000):
            prevNetG = "checkpoint/netG/netG_epoch_" + str(i) + ".pth"
            netG.load_state_dict(torch.load(prevNetG))
            print("Calculating FID...")
            real_sample = next(iter(test_dataloader))
            test_samples = real_sample[0]
            print("Loaded {0} real test samples".format(fid_score_batch_size))
            test_noise = torch.randn(fid_score_batch_size, nz, 1, 1, device=device)
            gen_samples = netG(test_noise, one_hot(real_sample[1]))
            print("Generated {0} fake samples".format(fid_score_batch_size))
            print(test_samples.size())
            print(gen_samples.size())
            test_samples.to(device)
            gen_samples.to(device)
            score = calc_fid(test_samples.detach(), gen_samples.detach(), inception)
            with open(fid_file, "a+") as f:
              f.write("{0}\n".format(score))
            del test_samples, gen_samples
            gc.collect()

# use_gpu = torch.cuda.is_available()
# device = torch.device("cuda" if use_gpu else "cpu")
# gan = SAGAN(ngpu=1)
# inception = load_inception(use_gpu)
# calc_fid_all_epochs(gan.netG, inception, device)

#labels = torch.zeros(10).long().random_(0, num_classes)
#print("labels:", labels)
#oh = one_hot(labels)
#print("one hot labels:", oh)
#print("one hot size", oh.size())

