# ------------------------------------------------------------------------
# Copyright (c) 2022 Woon-Ha Yeo <woonhahaha@gmail.com>.
# Copyright (c) 2022 Wang-Taek Oh <mm0741@naver.com>.
# ------------------------------------------------------------------------

"""
CAIR: Multi-Scale Color Attention Network for Instagram Filter Removal
+ Ensemble Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.arch_util import get_gaussian_kernel, Upsample
from basicsr.models.archs.NAFNet_arch import NAFBlock
from basicsr.models.archs.CAIR_arch import ColorAttention


class EnsembleNet(nn.Module):
    def __init__(self, img_channel=3, width=32, ens_num=3, blk_num=3, tta=False):
        super().__init__()
        self.tta = tta
        self.intro = nn.Conv2d(
            in_channels=img_channel * ens_num,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )
        self.ensemble = nn.Sequential(*[NAFBlock(width) for _ in range(blk_num)])
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

        self.device = torch.device("cuda")
        self.precision = "single"

    def forward_basic(self, inp):
        x = self.intro(inp)
        x = self.ensemble(x)
        x = self.ending(x)
        return x

    def forward(self, x):
        self.to(self.device)
        if self.tta:
            return self.forward_x8(x, forward_function=self.forward_basic)
        else:
            return self.forward_basic(x.to(self.device))

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != "single":
                v = v.float()

            v2np = v.data.cpu().numpy()
            if op == "v":
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == "h":
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == "t":
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == "half":
                ret = ret.half()

            return ret

        lr_list = [x.to(self.device)]
        for tf in "v", "h", "t":
            lr_list.extend([_transform(t, tf) for t in lr_list])
        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], "t")
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], "h")
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], "v")

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output


if __name__ == "__main__":
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

    ens_num = 3
    blk_num = 3

    print("ens_num", ens_num, "blk_num", blk_num, "width", width)

    using("start . ")
    net = EnsembleNet(img_channel=img_channel, width=width, blk_num=blk_num, tta=True)

    using("network .. ")

    # for n, p in net.named_parameters()
    #     print(n, p.shape)

    inp = torch.randn((1, 9, 256, 256))

    out = net(inp)
    final_mem = using("end .. ")
    # out.sum().backward()

    # out.sum().backward()

    # using('backward .. ')

    # exit(0)

    inp_shape = (9, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(
        net, inp_shape, as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)

    print("total .. ", params * 8 + final_mem)
