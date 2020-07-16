import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(
            nn.Conv2d(3 + c_dim,
                      conv_dim,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False))
        layers.append(
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(
                nn.Conv2d(curr_dim,
                          curr_dim * 2,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False))
            layers.append(
                nn.InstanceNorm2d(curr_dim * 2,
                                  affine=True,
                                  track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(
                nn.ConvTranspose2d(curr_dim,
                                   curr_dim // 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False))
            layers.append(
                nn.InstanceNorm2d(curr_dim // 2,
                                  affine=True,
                                  track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(
            nn.Conv2d(curr_dim,
                      3,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Generator_dissection(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator_dissection, self).__init__()

        self.conv1 = nn.Conv2d(3 + c_dim,
                               conv_dim,
                               kernel_size=7,
                               stride=1,
                               padding=3,
                               bias=False)
        self.norm1 = nn.InstanceNorm2d(conv_dim,
                                       affine=True,
                                       track_running_stats=True)
        self.ac1 = nn.ReLU(inplace=True)

        # Down-sampling layers.
        curr_dim = conv_dim

        self.conv2_1 = nn.Conv2d(curr_dim,
                                 curr_dim * 2,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1,
                                 bias=False)
        self.norm2_1 = nn.InstanceNorm2d(curr_dim * 2,
                                         affine=True,
                                         track_running_stats=True)
        self.ac2_1 = nn.ReLU(inplace=True)
        curr_dim = curr_dim * 2
        self.conv2_2 = nn.Conv2d(curr_dim,
                                 curr_dim * 2,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1,
                                 bias=False)
        self.norm2_2 = nn.InstanceNorm2d(curr_dim * 2,
                                         affine=True,
                                         track_running_stats=True)
        self.ac2_2 = nn.ReLU(inplace=True)
        curr_dim = curr_dim * 2

        # Bottleneck layers.
        self.res1 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.res2 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.res3 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.res4 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.res5 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)
        self.res6 = ResidualBlock(dim_in=curr_dim, dim_out=curr_dim)

        # Up-sampling layers.
        self.conv3_1 = nn.ConvTranspose2d(curr_dim,
                                          curr_dim // 2,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        self.norm3_1 = nn.InstanceNorm2d(curr_dim // 2,
                                         affine=True,
                                         track_running_stats=True)
        self.ac3_1 = nn.ReLU(inplace=True)
        curr_dim = curr_dim // 2
        self.conv3_2 = nn.ConvTranspose2d(curr_dim,
                                          curr_dim // 2,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        self.norm3_2 = nn.InstanceNorm2d(curr_dim // 2,
                                         affine=True,
                                         track_running_stats=True)
        self.ac3_2 = nn.ReLU(inplace=True)
        curr_dim = curr_dim // 2

        self.conv4 = nn.Conv2d(curr_dim,
                               3,
                               kernel_size=7,
                               stride=1,
                               padding=3,
                               bias=False)
        self.ac4 = nn.Tanh()

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.ac1(x)

        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = self.ac2_1(x)
        x = self.conv2_2(x)
        x = self.norm2_2(x)
        x = self.ac2_2(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.conv3_1(x)
        x = self.norm3_1(x)
        x = self.ac3_1(x)
        x = self.conv3_2(x)
        x = self.norm3_2(x)
        x = self.ac3_2(x)

        x = self.conv4(x)
        out = self.ac4(x)

        return out

    def h1(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.ac1(x)

        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = self.ac2_1(x)
        x = self.conv2_2(x)
        x = self.norm2_2(x)
        x = self.ac2_2(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.conv3_1(x)
        out = self.norm3_1(x)

        return out

    def f1(self, x):
        x = self.ac3_1(x)
        x = self.conv3_2(x)
        x = self.norm3_2(x)
        x = self.ac3_2(x)

        x = self.conv4(x)
        out = self.ac4(x)

        return out

    def h2(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.ac1(x)

        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = self.ac2_1(x)
        x = self.conv2_2(x)
        x = self.norm2_2(x)
        x = self.ac2_2(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.conv3_1(x)
        x = self.norm3_1(x)
        x = self.ac3_1(x)
        x = self.conv3_2(x)
        out = self.norm3_2(x)

        return out

    def f2(self, x):
        x = self.ac3_2(x)
        x = self.conv4(x)
        out = self.ac4(x)

        return out

    def h0(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.ac1(x)

        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = self.ac2_1(x)
        x = self.conv2_2(x)
        x = self.norm2_2(x)
        x = self.ac2_2(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        out = self.res6(x)

        return out

    def f0(self, x):
        x = self.conv3_1(x)
        x = self.norm3_1(x)
        x = self.ac3_1(x)
        x = self.conv3_2(x)
        x = self.norm3_2(x)

        x = self.ac3_2(x)
        x = self.conv4(x)
        out = self.ac4(x)

        return out

    def h01(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.ac1(x)

        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = self.ac2_1(x)
        x = self.conv2_2(x)
        x = self.norm2_2(x)
        x = self.ac2_2(x)

        out = self.res1(x)

        return out

    def f01(self, x):
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.conv3_1(x)
        x = self.norm3_1(x)
        x = self.ac3_1(x)
        x = self.conv3_2(x)
        x = self.norm3_2(x)

        x = self.ac3_2(x)
        x = self.conv4(x)
        out = self.ac4(x)

        return out

    def h02(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.ac1(x)

        x = self.conv2_1(x)
        x = self.norm2_1(x)
        x = self.ac2_1(x)
        x = self.conv2_2(x)
        out = self.norm2_2(x)

        return out

    def f02(self, x):
        x = self.ac2_2(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.conv3_1(x)
        x = self.norm3_1(x)
        x = self.ac3_1(x)
        x = self.conv3_2(x)
        x = self.norm3_2(x)

        x = self.ac3_2(x)
        x = self.conv4(x)
        out = self.ac4(x)

        return out

    def h03(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.ac1(x)

        x = self.conv2_1(x)
        out = self.norm2_1(x)

        return out

    def f03(self, x):
        x = self.ac2_1(x)
        x = self.conv2_2(x)
        x = self.norm2_2(x)
        x = self.ac2_2(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)

        x = self.conv3_1(x)
        x = self.norm3_1(x)
        x = self.ac3_1(x)
        x = self.conv3_2(x)
        x = self.norm3_2(x)

        x = self.ac3_2(x)
        x = self.conv4(x)
        out = self.ac4(x)

        return out


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(
                nn.Conv2d(curr_dim,
                          curr_dim * 2,
                          kernel_size=4,
                          stride=2,
                          padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim,
                               1,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(curr_dim,
                               c_dim,
                               kernel_size=kernel_size,
                               bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
