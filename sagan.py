import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data


class SAGAN:

    nz =100 
    ngf = 64 
    ndf = 64 
    nc = 3
    ngpu = 1

    def __init__(self, nc=3, nz=100, ngf=64, ndf=64, ngpu=1):
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.nc = nc
        self.ngpu = ngpu
        self.netD = _Discriminator(ngpu=ngpu, nc=nc, ndf=ndf)
        self.netD.apply(self._weights_init)
        self.netG = _Generator(ngpu=ngpu, nc=nc, nz=nz, ngf=ngf)
        self.netG.apply(self._weights_init)
        self.loss = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # custom weights initialization called on netG and netD
    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def train_on_batch(self, data, device):
        real_label = 1
        fake_label = 0
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        self.netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)
    
        output = self.netD(real_cpu)
        errD_real = self.loss(output, label)
        errD_real.backward()
        D_x = output.mean().item()
    
        # train with fake
        noise = torch.randn(batch_size, self.nz, 1, 1, device=device)
        fake = self.netG(noise)
        label.fill_(fake_label)
        output = self.netD(fake.detach())
        errD_fake = self.loss(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        self.optimizerD.step()
    
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = self.netD(fake)
        errG = self.loss(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizerG.step()

        print('Loss_D: %.4f Loss_G: %.4f' % (errD.item(), errG.item(),))

#nn.utils.spectral_norm(nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False))

class _Generator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf):
        super(_Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            #Buffer(ngf * 8),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(_Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
