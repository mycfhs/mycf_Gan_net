from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from net import _netD, _netG

import metric
from metric import make_dataset
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')

    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    ########################################################
    #### For evaluation ####
    parser.add_argument('--sampleSize', type=int, default=2000, help='number of samples for evaluation')
    ########################################################

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    #########################
    #### Dataset prepare ####
    #########################
    dataset = make_dataset(dataset=opt.dataset, dataroot=opt.dataroot, imageSize=opt.imageSize)
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    #########################
    #### Models building ####
    #########################
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)

    netG = _netG().to(device)
    # netG.apply(weights_init)
    # if opt.netG != '':
    #     netG.load_state_dict(torch.load(opt.netG))
    score_tr = np.zeros((61, 4 * 7 + 3))
    incep, modescore, fid = [], [], []

    for model_epoch in tqdm(range(1, 62)):
        netG.load_state_dict(torch.load(r'C:\Users\22862\Desktop\Gan_net\gan_2img\netG_epoch_%d.pth'%model_epoch))

        netD = _netD().to(device)
        # netD.apply(weights_init)
        # if opt.netD != '':
        #     netD.load_state_dict(torch.load(opt.netD))
        netD.load_state_dict(torch.load(r'C:\Users\22862\Desktop\Gan_net\gan_2img\netD_epoch_%d.pth'%model_epoch))

        criterion = nn.BCELoss()

        fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
        real_label = 1
        fake_label = 0

        # setup optimizer
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # [emd-mmd-knn(knn,real,fake,precision,recall)]*4 - IS - mode_score - FID

        # compute initial score
        s = metric.compute_score_raw(opt.dataset, opt.imageSize, opt.dataroot, opt.sampleSize, 16, opt.outf+'/real/', opt.outf+'/fake/',
                                     netG, opt.nz, conv_model='inception_v3', workers=int(opt.workers))
        incep.append(s[-3])
        modescore.append(s[-2])
        fid.append(s[-1])
        # score_tr[0] = s
        # print(score_tr)
        # np.save('%s/score_tr.npy' % (opt.outf), score_tr)

    import matplotlib.pyplot as plt
    epochl = [i+1 for i in range(len(incep))]
    plt.subplot(3,1,1)
    plt.plot(epochl, incep, color='b', label='incep')
    plt.title('incep')
    plt.subplot(3, 1, 2)
    plt.plot(epochl, modescore, color='r', label='modescore')
    plt.title('modescore')
    plt.subplot(3, 1, 3)
    plt.plot(epochl, fid, color='m', label='fid')
    plt.title('fid')
    plt.show()
