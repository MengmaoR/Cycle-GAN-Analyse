import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_nc):
        super(SelfAttention, self).__init__()
        
        # f, g, h 三个卷积，kernel_size=1，用于生成Q, K, V
        self.f = nn.Conv2d(input_nc, input_nc // 8, kernel_size=1)  # Query
        self.g = nn.Conv2d(input_nc, input_nc // 8, kernel_size=1)  # Key
        self.h = nn.Conv2d(input_nc, input_nc // 2, kernel_size=1)  # Value
        
        # 最终映射，将加权后的 h(x) 通道数映射回 input_nc
        self.v = nn.Conv2d(input_nc // 2, input_nc, kernel_size=1)
        
        # 用于可学习的缩放
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        
        # f(x) -> Query
        f_out = self.f(x)  # shape = (B, C//8, H, W)
        # g(x) -> Key
        g_out = self.g(x)  # shape = (B, C//8, H, W)
        # h(x) -> Value
        h_out = self.h(x)  # shape = (B, C//2, H, W)
        
        # 调整形状，方便做矩阵乘法
        # (B, C//8, H*W)
        f_out = f_out.view(B, -1, H*W)  # Query
        # (B, C//8, H*W)
        g_out = g_out.view(B, -1, H*W)  # Key
        # (B, C//2, H*W)
        h_out = h_out.view(B, -1, H*W)  # Value
        
        # 计算注意力分数 (B, N, N)，其中 N = H*W
        # f_out: (B, C_q, N), g_out: (B, C_q, N)
        # => (B, N, N)
        attn_score = torch.bmm(f_out.permute(0, 2, 1), g_out)  # (B, N, N)
        # 归一化
        attn = F.softmax(attn_score, dim=-1)  # (B, N, N)
        
        # 用注意力加权 h_out (Value)
        # h_out: (B, C_v, N)
        # attn: (B, N, N)
        # => (B, C_v, N)
        attn_h = torch.bmm(h_out, attn.permute(0, 2, 1))
        
        # 调整回 (B, C_v, H, W)
        attn_h = attn_h.view(B, -1, H, W)
        
        # 映射回 input_nc，并加上原输入 (skip connection)
        out = self.v(attn_h)
        out = x + self.gamma * out  # gamma是可学习的缩放系数
        
        # 返回输出和注意力矩阵
        return out, attn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, use_attention=False):
        super(ResidualBlock, self).__init__()
        self.use_attention = use_attention

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

        if use_attention:
            self.attn = SelfAttention(in_features)
        else:
            self.attn = None
            

    def forward(self, x):
        out = x + self.conv_block(x)
        if self.use_attention:
            out, attn = self.attn(out)
            return out, attn
        else:
            return out

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # 降采样
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for i in range(n_residual_blocks):
            if i == 4:  # 在第5个残差块后加入自注意力模块
                model += [ResidualBlock(in_features, use_attention=True)]
            else:
                model += [ResidualBlock(in_features)]

        # 升采样
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # 输出层
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        attn_maps = []
        for layer in self.model:
            if isinstance(layer, ResidualBlock) and layer.use_attention:
                x, attn = layer(x)
                attn_maps.append(attn)
            else:
                x = layer(x)
                
        return x, attn_maps

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
        # (252 - 4 + 2*1)/1 + 1 = 251
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