import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# rfsize = f(out, stride, ksize) = (out - 1) * stride + ksize
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        x = F.avg_pool2d(x, x.size()[2:])
        return x.view(x.size()[0], -1)

class PatchGAN16(nn.Module):
    def __init__(self, input_nc):
        super(PatchGAN16, self).__init__()

        # (13-1)*1 + 4 = 16
        # (256 - 4 + 2*1)/1 + 1 = 255
        model = [   nn.Conv2d(input_nc, 64, 4, stride=1, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # (10-1)*1 + 4 = 13
        # (255 - 4 + 2*1)/1 + 1 = 254
        model += [  nn.Conv2d(64, 128, 4, stride=1, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # (7-1)*1 + 4 = 10
        # (254 - 4 + 2*1)/1 + 1 = 253
        model += [  nn.Conv2d(128, 256, 4, stride=1, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # (4-1)*1 + 4 = 7
        # (253 - 4 + 2*1)/1 + 1 = 252
        model += [  nn.Conv2d(256, 512, 4, stride=1, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # (1-1)*1 + 4 = 4
        # (252 - 4 + 2*1)/1 + 1 = 253
        model += [nn.Conv2d(512, 1, 4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class PatchGAN70(nn.Module):
    def __init__(self, input_nc):
        super(PatchGAN70, self).__init__()

        # (34-1)*2 + 4 = 70
        # (256 - 4 + 2*1)/2 + 1 = 128
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # (16-1)*2 + 4 = 34
        # (128 - 4 + 2*1)/2 + 1 = 64
        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # (7-1)*2 + 4 = 16
        # (64 - 4 + 2*1)/2 + 1 = 32
        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # (4-1)*1 + 4 = 7
        # (32 - 4 + 2*1)/1 + 1 = 31
        model += [  nn.Conv2d(256, 512, 4, stride=1, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # (1-1)*1 + 4 = 4
        # (31 - 4 + 2*1)/1 + 1 = 30
        model += [nn.Conv2d(512, 1, 4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class PatchGAN142(nn.Module):
    def __init__(self, input_nc):
        super(PatchGAN142, self).__init__()

        # (70-1)*2 + 4 = 142
        # (256 - 4 + 2*1)/2 + 1 = 128
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # (34-1)*2 + 4 = 70
        # (128 - 4 + 2*1)/2 + 1 = 64
        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # (16-1)*2 + 4 = 34
        # (64 - 4 + 2*1)/2 + 1 = 32
        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]
        
        # (7-1)*2 + 4 = 16
        # (32 - 4 + 2*1)/2 + 1 = 16
        model += [  nn.Conv2d(256, 512, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # (4-1)*1 + 4 = 7
        # (16 - 4 + 2*1)/1 + 1 = 15
        model += [  nn.Conv2d(512, 512, 4, stride=1, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # (1-1)*1 + 4 = 4
        # (15 - 4 + 2*1)/1 + 1 = 14
        model += [nn.Conv2d(512, 1, 4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)