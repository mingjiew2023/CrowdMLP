import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
import collections
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import  math
import numpy as np
from Transformer import ViT

class CSRNet(nn.Module):
    def __init__(self, load_weights=False,traininng=True): #ad_weights=False
        super(CSRNet, self).__init__()
        self.seen = 0
        self.training = traininng
        self.frontend_feat_0 = [64, 64, 'M', 128, 128, 'M', ]
        self.frontend_0 = make_layers_new(self.frontend_feat_0)
        self.frontend_feat_1 = [256, 256, 256, 'M', 512, 512, 512]
        self.frontend_1 = make_layers_new(self.frontend_feat_1, in_channels=128)

        self.backend_feat = [128]
        self.backend = make_layers_new(self.backend_feat, in_channels=512, dilation=False)


        self.forward_mlpmixer1 = MLPMixer(image_size=22, patch_size=2,num_patches=340,dim=256, depth=3,num_classes=4)  #121

        self.head1 = MLPMixer(image_size=22, patch_size=2, num_patches=64,dim=256, depth=1,num_classes=4)
        self.head2 = MLPMixer(image_size=22, patch_size=2, num_patches=16, dim=256, depth=1, num_classes=4)
        self.head_raw = MLPMixer(image_size=22, patch_size=2, num_patches=256, dim=256, depth=1, num_classes=4)
        self.head_global = MLPMixer(image_size=22, patch_size=2, num_patches=4, dim=256, depth=1, num_classes=4)


        self.count_head =  nn.Sequential(nn.Conv1d(in_channels=340, out_channels=484, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(484),
                                         nn.Dropout2d(0.5),
                                         nn.Linear(256,1),
                                         nn.ReLU()    #remove for ShanghaiB
        )

        self.token = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 4, p2 = 4),
                    nn.Linear((128*4 ** 2) , 256)) #64

        self.token1 = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1= 8, p2= 8),
                                    nn.Linear((128 * 8 ** 2), 256),) #16
        self.token2 = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16),
                                   nn.Linear((128 * 16 ** 2), 256), ) #4

        self.token_img = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16),
                                    nn.Linear((3 * 16 ** 2), 256), #256
                                    nn.Dropout2d(0.2))

        if not load_weights:
            mod = models.vgg16_bn(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend_0.state_dict().items())):
                temp_key = list(self.frontend_0.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend_0.load_state_dict(fsd)

            fsd = collections.OrderedDict()
            for i in range(len(self.frontend_1.state_dict().items())):
                temp_key = list(self.frontend_1.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i+len(self.frontend_0.state_dict().items())][1]
            self.frontend_1.load_state_dict(fsd)

    def forward(self, x):
        x_ori = x
        x_vgg_0 = self.frontend_0(x)
        x_vgg = self.frontend_1(x_vgg_0)
        x = self.backend(x_vgg)

        x_token = self.token(x)
        x_token1 = self.token1(x)
        x_token_global = self.token2(x)
        x_token_img = self.token_img(x_ori)

        #multi-head mlpmixer
        x_token = self.head1(x_token)
        x_token1 = self.head2(x_token1)
        x_token_global =self.head_global(x_token_global)
        x_token_img = self.head_raw(x_token_img)

        x_token = torch.cat((x_token_img, x_token, x_token1,x_token_global), dim=1)

        x_map_ori = self.forward_mlpmixer1(x_token)

        x_num = self.count_head(x_map_ori).sum(dim=1)

        return x_num  #,self.count_head(x_map_ori)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(x) + x #self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear,num_patch=0):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.ReLU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.ReLU(),
        #nn.BatchNorm1d(num_patch),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, patch_size, dim, depth,num_patches,in_channel=128, num_classes, expansion_factor = 4, dropout = 0.2):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = num_patches#(image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
    *[nn.Sequential(
    PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first,num_patch=num_patches)),
    nn.BatchNorm1d(num_patches),
    PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last,num_patch=num_patches)),
                nn.BatchNorm1d(num_patches)
            ) for _ in range(depth)],
    )



def make_layers_new(cfg, in_channels=3, batch_norm=True, dilation=False,dilation_rate=2):
    if dilation:
        d_rate = dilation_rate
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers(cfg, in_channels=3, batch_norm=True, dilation=False,d_rate=2):
    if dilation:
        d_rate = d_rate
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d,nn.ReLU(inplace=True),nn.GroupNorm(num_groups=16,num_channels=v)]  #,nn.GroupNorm(num_groups=4,num_channels=v)
            in_channels = v
    return nn.Sequential(*layers)


