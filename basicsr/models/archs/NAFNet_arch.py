# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.arch_util import MySequential
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.arch_util import get_gaussian_kernel, Upsample

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class NAFGroup(nn.Module):
    def __init__(self, n_feats, num_nb=2):
        super().__init__()
        self.body = nn.Sequential(
                        *[NAFBlock(n_feats) for _ in range(num_nb)]
                    )
        self.tail = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, inp):
        x = inp

        x = self.body(x)
        x = self.tail(x)
        return x + inp


class ColorAttention(nn.Module):
    def __init__(self, in_channel, num_feat):
        super(ColorAttention, self).__init__()        
        output_nc = num_feat

        num_ng = 2
        num_nb = 2
        kernel_size = 1

        sigma = 12 ## GAUSSIAN_SIGMA

        modules_head = [nn.Conv2d(in_channel, num_feat, kernel_size=kernel_size, padding=0, stride=1, groups=1, bias=True)]

        modules_downsample = [nn.MaxPool2d(kernel_size=2)] 
        self.downsample = nn.Sequential(*modules_downsample)

        modules_body = [
                NAFGroup(num_feat, num_nb) \
                for _ in range(num_ng)
            ]
        
        modules_body.append(nn.Conv2d(num_feat, num_feat, kernel_size, padding=0, stride=1, groups=1, bias=True))

        modules_tail = [nn.Conv2d(num_feat, output_nc, kernel_size, padding=0, stride=1, groups=1, bias=True), nn.Sigmoid()]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.blur, self.pad = get_gaussian_kernel(sigma=sigma)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        x = self.blur(x)
        x = self.head(x)
        x = self.downsample(x)  
        x = self.body(x)
        x = self.tail(x)
        return x


class ColorNAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        self.color_attention = ColorAttention(in_channel=img_channel, num_feat=width)
        self.upsample = Upsample(scale=2, num_feat=width)
        self.conv1x1 = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        
        color = self.color_attention(inp)
        color = self.upsample(color)
        color = self.conv1x1(color)
        color = x * color
        x = x + color

        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class ColorMISONet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro1 = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.intro2 = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.intro3 = nn.Conv2d(in_channels=img_channel, out_channels=width*2, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.intro4 = nn.Conv2d(in_channels=img_channel, out_channels=width*4, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()


        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, chan, 2, 2)
            )
            chan = chan * 2
        
        chan = chan // 2
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)],
                nn.Conv2d(chan, chan * 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
            )
        
        chan = chan * 2
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            

        self.padder_size = 2 ** len(self.encoders)

        self.color_attention1 = ColorAttention(in_channel=img_channel, num_feat=width)
        self.color_attention2 = ColorAttention(in_channel=img_channel, num_feat=width)
        self.color_attention3 = ColorAttention(in_channel=img_channel, num_feat=width*2)
        self.color_attention4 = ColorAttention(in_channel=img_channel, num_feat=width*4)
        self.upsample = Upsample(scale=2, num_feat=width)
        self.conv1x1 = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, inp):
        B, C, H, W = inp.shape
        datas = []
        inp = self.check_image_size(inp)
        datas.append(self.intro1(inp))

        inp_2_color = self.color_attention2(inp)
        inp_2_resize = F.interpolate(inp, scale_factor=0.5, mode='bicubic')
        inp_2_temp = self.intro2(inp_2_resize)
        inp_2_color = inp_2_temp * inp_2_color
        inp_2_color = inp_2_temp + inp_2_color
        datas.append(inp_2_color)

        inp_4_color = self.color_attention3(inp_2_resize)
        inp_4_resize = F.interpolate(inp_2_resize, scale_factor=0.5, mode='bicubic')
        inp_4_temp = self.intro3(inp_4_resize)
        inp_4_color = inp_4_temp * inp_4_color
        inp_4_color = inp_4_temp + inp_4_color
        datas.append(inp_4_color)

        inp_8_color = self.color_attention4(inp_4_resize)
        inp_8_resize = F.interpolate(inp_4_resize, scale_factor=0.5, mode='bicubic')
        inp_8_temp = self.intro4(inp_8_resize)
        inp_8_color = inp_8_temp * inp_8_color
        inp_8_color = inp_8_temp + inp_8_color
        datas.append(inp_8_color)


        encs = []
        num = [0, 1, 2, 3]
        for encoder, down, data, i in zip(self.encoders, self.downs, datas, num):
            if i == 0:
                x = data
            else:
                x = torch.cat((x, data), 1)
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        
        color = self.color_attention1(inp)
        color = self.upsample(color)
        color = self.conv1x1(color)
        color = x * color
        x = x + color

        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class ColorNAFNetLocal(Local_Base, ColorNAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        ColorNAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class ColorMISONAFNetLocal(Local_Base, ColorMISONet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        ColorMISONet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x+factor*(new_x-x) for x, new_x in zip(feats, new_feats)])
        return new_feats
    

class LAM_Module(nn.Module):
    '''
    Layer attention module from 
    https://github.com/wwlCape/HAN/blob/master/src/model/han.py
    '''
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self, x):
        '''
        inputs :
            x : input feature maps( B X N X C X H X W)
        returns :
            out : attention value + input feature
            attention: B X N X N
        '''
        if len(x.size()) == 4:
            N, C, height, width = x.size()
            m_batchsize = 1
        else:
            m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma * out + x
        out = out.view(m_batchsize, -1, height, width)
        return out
    
    
class NAFL(nn.Module):
    '''
    NAFNet + Layer attention for Image Restoration
    (Same with NAFLSR architecture from NAFSSR_arch.py except Pixelshuffle layer)
    '''
    def __init__(self, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0.):
        super().__init__()
        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True
        )
        self.body = MySequential(
            *[DropPath(
                drop_path_rate, 
                NAFBlock(
                    width,
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        )
        
        self.inp_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=img_channel,
                out_channels=img_channel,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=1,
                bias=True
            ),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=width,
                out_channels=img_channel,
                kernel_size=3,
                padding=1, 
                stride=1,
                groups=1,
                bias=True
            ),
        )
        self.conv1 = nn.Conv2d(
                        in_channels=num_blks * width,
                        out_channels=width,
                        kernel_size=1
                    )
        self.la = LAM_Module(width)
        
    def forward(self, inp):
        inp_feat = self.inp_conv(inp)
        feats = self.intro(inp)
        
        for name, midlayer in self.body._modules.items():
            feats = midlayer(feats)[0]
            if name == '0':
                feats_cat = feats.unsqueeze(1)
            else:
                feats_cat = torch.cat([feats.unsqueeze(1), feats_cat], 1)
        nf_out = feats
        lam_out = self.la(feats_cat)
        lam_out = self.conv1(lam_out)
        
        out = nf_out + lam_out
        out = self.out_conv(out)
        out = out + inp_feat
        return out


if __name__ == '__main__':
    import resource
    def using(point=""):
        # print(f'using .. {point}')
        usage = resource.getrusage(resource.RUSAGE_SELF)
        global Total, LastMem

        # if usage[2]/1024.0 - LastMem > 0.01:
        # print(point, usage[2]/1024.0)
        print(point, usage[2] / 1024.0)

        LastMem = usage[2] / 1024.0
        return usage[2] / 1024.0

    img_channel = 3
    width = 32
    
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]
    
    print('enc blks', enc_blks, 'middle blk num', middle_blk_num, 'dec blks', dec_blks, 'width' , width)
    
    using('start . ')
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, 
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    using('network .. ')

    # for n, p in net.named_parameters()
    #     print(n, p.shape)


    inp = torch.randn((1, 3, 256, 256))

    out = net(inp)
    final_mem = using('end .. ')
    # out.sum().backward()

    # out.sum().backward()

    # using('backward .. ')

    # exit(0)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)

    print('total .. ', params * 8 + final_mem)