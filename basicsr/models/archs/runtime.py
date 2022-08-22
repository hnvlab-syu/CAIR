import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from CAIR_arch import *
from NAFNet_arch import NAFNet, NAFNetLocal
from CIFR_arch import CIFR_Encoder, Discriminator, PatchDiscriminator
from IFRNet_arch import IFRNet
from Ensemble_arch import EnsembleNet

model_dict = {
    'ifrnet': [
        IFRNet(32, 32),
        models.vgg16(pretrained=True).features.eval(),
        Discriminator(32), PatchDiscriminator(32)
    ],
    'cifr': [
        CIFR_Encoder(32, 32),
        models.vgg16(pretrained=True).features.eval(),
        Discriminator(32), PatchDiscriminator(32)
    ]
}
for model in model_dict.keys():
    print(model)
    for _ in range(10):
        main = model_dict[model][0].cpu()
        vgg16 = model_dict[model][1].cpu()
        disc = model_dict[model][2].cpu()
        pdisc = model_dict[model][3].cpu()
        inputs = torch.randn(1, 3, 256, 256).cpu()
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                vgg_feat = vgg16(inputs)
                main(inputs, vgg_feat)
                disc(inputs)
                pdisc(inputs)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))


        main = main.cuda()
        vgg16 = vgg16.cuda()
        disc = disc.cuda()
        pdisc = pdisc.cuda()
        inputs = torch.randn(1, 3, 256, 256).cuda()
        with profile(activities=[
                ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                vgg_feat = vgg16(inputs)
                main(inputs, vgg_feat)
                disc(inputs)
                pdisc(inputs)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))