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
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.arch_util import get_gaussian_kernel, Upsample
from basicsr.models.archs.NAFNet_arch import NAFBlock
from basicsr.models.archs.CAIR_arch import ColorAttention


class CAIR_S_TTA(nn.Module):
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

        self.device = torch.device('cuda')
        self.precision = 'single'

    def forward_function(self, inp):
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
    
    def forward(self, x):
        self.to(self.device)
        return self.forward_x8(x, forward_function=self.forward_function)
    
    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x.to(self.device)]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class CAIR_M_TTA(nn.Module):
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

        self.device = torch.device('cuda')
        self.precision = 'single'

    def forward_function(self, inp):
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
    
    def forward(self, x):
        self.to(self.device)
        return self.forward_x8(x, forward_function=self.forward_function)
    
    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x.to(self.device)]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class CAIR_S_TTA_Local(Local_Base, CAIR_S_TTA):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        CAIR_S_TTA.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class CAIR_M_TTA_Local(Local_Base, CAIR_M_TTA):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        CAIR_M_TTA.__init__(self, *args, **kwargs)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


class EnsembleNet(nn.Module):
    def __init__(self, img_channel=3, width=32, ens_num=3, blk_num=3):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel * ens_num, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ensemble = nn.Sequential(
                        *[NAFBlock(width) for _ in range(blk_num)]
                    )
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

    def forward(self, inp):
        x = self.intro(inp)
        x = self.ensemble(x)
        x = self.ending(x)
        return x


class EnsembleNetTTA(nn.Module):
    def __init__(self, img_channel=3, width=32, ens_num=3, blk_num=3):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel * ens_num, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ensemble = nn.Sequential(
                        *[NAFBlock(width) for _ in range(blk_num)]
                    )
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.device = torch.device('cuda')
        self.precision = 'single'
        
    def forward_function(self, inp):
        x = self.intro(inp)
        x = self.ensemble(x)
        x = self.ending(x)
        return x

    def forward(self, x):
        self.to(self.device)
        return self.forward_x8(x, forward_function=self.forward_function)
    
    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x.to(self.device)]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output


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
    net = EnsembleNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, 
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