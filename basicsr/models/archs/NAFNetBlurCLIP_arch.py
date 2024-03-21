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
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.KernelNet_arch import BlurEncoder, Bottleneck
from einops.layers.torch import Rearrange

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

class NAFBlockHyper(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()

        dw_channel = c * DW_Expand
        self.dw_channel = dw_channel
        #self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        #self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        #self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        """
        self.w1 = nn.Parameter(torch.randn(dw_channel, c, 1, 1), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(dw_channel), requires_grad=True)
        nn.init.kaiming_uniform_(self.w1, mode='fan_in', nonlinearity='relu')
        
        self.w2 = nn.Parameter(torch.randn(dw_channel, 1, 3, 3), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(dw_channel))
        nn.init.kaiming_uniform_(self.w2, mode='fan_in', nonlinearity='relu')
        

        self.w3 = nn.Parameter(torch.randn(c, dw_channel//2, 1, 1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(c))
        nn.init.kaiming_uniform_(self.w3, mode='fan_in', nonlinearity='relu')
        
        self.w4 = nn.Parameter(torch.randn(ffn_channel, c, 1, 1), requires_grad=True)
        self.b4 = nn.Parameter(torch.zeros(ffn_channel), requires_grad=True)
        nn.init.kaiming_uniform_(self.w4, mode='fan_in', nonlinearity='relu')
        
        self.w5 = nn.Parameter(torch.randn(c, ffn_channel//2, 1, 1), requires_grad=True)
        self.b5 = nn.Parameter(torch.zeros(c), requires_grad=True)
        nn.init.kaiming_uniform_(self.w5, mode='fan_in', nonlinearity='relu')
        """
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        #self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        #self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, weights_and_bias):
        x = inp
        B = x.shape[0]

        x = self.norm1(x)
        
        w1 = weights_and_bias['conv1']['weight']
        b1 = weights_and_bias['conv1']['bias']
        #import pdb; pdb.set_trace()
        #x = [F.conv2d(x[i:i+1], self.w1+w1[i], self.b1+b1[i], padding=0) for i in range(B)]
        x = [F.conv2d(x[i:i+1], w1[i], b1[i], padding=0) for i in range(B)]
        x = torch.cat(x, dim=0)
        
        w2 = weights_and_bias['conv2']['weight']
        b2 = weights_and_bias['conv2']['bias']
        #x = [F.conv2d(x[i:i+1], self.w2+w2[i], self.b2+b2[i], padding=1, groups=self.dw_channel) for i in range(B)]
        x = [F.conv2d(x[i:i+1], w2[i], b2[i], padding=1, groups=self.dw_channel) for i in range(B)]
        x = torch.cat(x, dim=0)
        
        x = self.sg(x)
        x = x * self.sca(x)
        
        w3 = weights_and_bias['conv3']['weight']
        b3 = weights_and_bias['conv3']['bias']
        #x = [F.conv2d(x[i:i+1], self.w3+w3[i], self.b3+b3[i], padding=0) for i in range(B)]
        x = [F.conv2d(x[i:i+1], w3[i], b3[i], padding=0) for i in range(B)]
        x = torch.cat(x, dim=0)
        

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)

        w4 = weights_and_bias['conv4']['weight']
        b4 = weights_and_bias['conv4']['bias']
        #x = [F.conv2d(x[i:i+1], self.w4+w4[i], self.b4+b4[i], padding=0) for i in range(B)]
        x = [F.conv2d(x[i:i+1], w4[i], b4[i], padding=0) for i in range(B)]
        x = torch.cat(x, dim=0)

        x = self.sg(x)

        w5 = weights_and_bias['conv5']['weight']
        b5 = weights_and_bias['conv5']['bias']
        #x = [F.conv2d(x[i:i+1], self.w5+w5[i], self.b5+b5[i], padding=0) for i in range(B)]
        x = [F.conv2d(x[i:i+1], w5[i], b5[i], padding=0) for i in range(B)]
        x = torch.cat(x, dim=0)

        x = self.dropout2(x)
        
        return y + x * self.gamma


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2,
                              padding_mode='reflect')
        self.norm = LayerNorm2d(num_out_layers)
        #nn.InstanceNorm2d(num_out_layers, track_running_stats=False, affine=True)

    def forward(self, x):
        return F.elu(self.norm(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)

class BlurEncoderDecoder(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, width=64): #heads, input_resolution=256,
        super().__init__()
        self.output_dim = output_dim
        #self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        self.conv_out = nn.Conv2d(width * 32, output_dim, 1, padding=0, bias=False)
        #embed_dim = width * 32  # the ResNet feature dimension
        #self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        self.attnpool = None
        

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)
    
    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.conv_out(x4)
        x = x.mean(-1).mean(-1)
        

        return x

class ResMLPModule(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.fc1 = nn.Linear(n_channels, n_channels)
        self.fc2 = nn.Linear(n_channels, n_channels)
        self.norm1 = nn.LayerNorm(n_channels)
        self.norm2 = nn.LayerNorm(n_channels)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):

        residual = x
        #x = F.gelu(x)
        x = self.norm1(self.fc1(x))
        x = F.relu(x)
        x = self.norm2(self.fc2(x))
        out = x + residual

        return out

class NAFNetBlurCLIP(nn.Module):

    def __init__(self, pretrained_clip_dir, crop_size, stride_crop, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
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

        self.pretrained_clip_dir = pretrained_clip_dir
        self.crop_size = crop_size
        self.stride_crop = stride_crop
        #self.b_encoder = BlurEncoderDecoder(layers=[3,4,6,3], output_dim=128, width=64)
        self.b_encoder = BlurEncoder(layers=[3,4,6,3], output_dim=128, width=64)
        self.load_pretrained_blurclip_parameters()

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

        self.padder_size = max(2 ** len(self.encoders), 32)

        self.conv_params_dict = OrderedDict()
        n_params= 0
        n_params0 = 0
        n_params1 = 0
        n_params2 = 0
        #or name.startswith('encoders.2')
        #name.startswith('decoders.2') or name.startswith('decoders.3')) and \
        for name, param in self.named_parameters(): 
            if (name.startswith('encoders.0') or name.startswith('encoders.1') or name.startswith('encoders.2')) and \
                (name.endswith('weight') or name.endswith('bias')) and \
                    name.find('conv')>0 and name.find('b_encoder')<0:
                dims=[-1]
                dims.extend(list(param.shape))
                #print(name, param.shape)
                self.conv_params_dict[name] = {'n_elements': param.nelement(), 'shape':dims}
                n_params = n_params+param.nelement()
                
                if name.startswith('encoders.0'):
                    n_params0 += param.nelement()
                if name.startswith('encoders.1'):
                    n_params1 += param.nelement()
                if name.startswith('encoders.2'):
                    n_params2 += param.nelement()

        print(f"##### HYPERNETWORK PARAMS #{str(n_params0)}, {str(n_params1)}, {str(n_params2)}, {str(n_params)}")
        
        #self.mlp_res_block1 = ResMLPModule(256)
        #self.mlp_res_block2 = ResMLPModule(256)
        
        #self.fc_hyper1 = nn.Linear(128, 256)
        #self.fc_hyper2_0 = nn.Linear(256, 512)
        #self.fc_hyper2_1 = nn.Linear(256, 512)
        #self.fc_hyper2_2 = nn.Linear(256, 512)
        #self.fc_hyper3 = nn.Linear(512, 1024)
        #self.fc_hyper4 = nn.Linear(1024, 2048)
        #self.fc_hyper3_0 = nn.Linear(512, n_params0)
        #self.fc_hyper3_1 = nn.Linear(512, n_params1)
        #self.fc_hyper3_2 = nn.Linear(512, n_params2)

        self.fc_hyper1 = nn.Linear(128, 256)
        self.fc_hyper2 = nn.Linear(256, 512)
        self.fc_hyper5 = nn.Linear(512, n_params)
    
        nn.init.kaiming_normal_(self.fc_hyper1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_hyper2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_hyper5.weight, mode='fan_out', nonlinearity='relu')

        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(512)

        # re-build Encoders.
        self.n_hyper = 3
        chan = width
        self.encoders_hyper = nn.ModuleList()
        for i, num in enumerate(enc_blk_nums):
            if i < self.n_hyper:
                assert num == 1
                self.encoders_hyper.append(
                    NAFBlockHyper(chan)
                )
                
                self.encoders.pop(0)
            chan = chan * 2
        """
        self.decoders_hyper = nn.ModuleList()
        count = 0
        for i, num in enumerate(dec_blk_nums):
            chan = chan //2
            if i < self.n_hyper:
                continue
            else:
                assert num == 1
                self.decoders_hyper.append(
                    NAFBlockHyper(chan)
                )
                self.decoders.pop(i-count)
                count=count+1
        """
        #for name, param in self.named_parameters(): 
            #if (name.startswith('encoders.0') or name.startswith('encoders.1') or name.startswith('encoders.2')) and \
            #    (name.endswith('weight') or name.endswith('bias')) and \
            #        name.find('conv')>0 and name.find('b_encoder')<0:
        #    print(name, param.shape)
        #import pdb; pdb.set_trace()
        
        #for name, param in self.b_encoder.named_parameters():

    def load_pretrained_blurclip_parameters(self):
        pretrained_dict = torch.load(self.pretrained_clip_dir)

        model_dict = self.b_encoder.state_dict()
        prefix = 'b_encoder.'
        
        pretrained_dict_tight = {}
        count = 0
        for k, v in pretrained_dict['params'].items():
            if k.startswith(prefix) and k[len(prefix):] in model_dict:
                count += 1
                pretrained_dict_tight[k[len(prefix):]] = v
        print(f"##### SUCESSFULLY LOADED #{str(count)} CLIP IMAGE ENCODER PARAMS")
        model_dict.update(pretrained_dict_tight)
        
        self.b_encoder.load_state_dict(model_dict)
        count = 0
        for name, param in self.b_encoder.named_parameters():
            #if name in pretrained_dict_tight:
            count+=1
            param.requires_grad = False
        print(f"##### SUCESSFULLY FREEZE #{str(count)} CLIP IMAGE ENCODER PARAMS")
        """
        for name, param in self.b_encoder.named_parameters(): 
            if param.requires_grad==False: 
                print(name)
        import pdb; pdb.set_trace()
        """

    #def feature_fusion(self, x):
    def grids(self, x):
        crop_size = self.crop_size
        stride_crop = self.stride_crop

        B, C, H, W, = x.shape

        n_H = (H - crop_size)//stride_crop + 1
        n_W = (W - crop_size)//stride_crop + 1
        
        
        x = F.unfold(x, kernel_size=(crop_size, crop_size), padding=(0, 0), stride=(stride_crop, stride_crop))
        x = Rearrange('b (c ph pw) (oh ow) -> (b oh ow) c ph pw', c=C, ph=crop_size, pw=crop_size, oh=n_H, ow=n_W)(x)
        
        # debug
        """
        x_identity = x
        y = Rearrange('(b oh ow) c ph pw -> b (c ph pw) (oh ow)', c=C, ph=crop_size, pw=crop_size, oh=n_H, ow=n_W)(x)
        counts = torch.ones_like(y)

        y = F.fold(y, output_size=(H, W) , kernel_size=(crop_size, crop_size), padding=(0, 0), stride=(stride_crop, stride_crop))
        counts = F.fold(counts, output_size=(H, W) , kernel_size=(crop_size, crop_size), padding=(0, 0), stride=(stride_crop, stride_crop))

        y = y / counts
        import pdb; pdb.set_trace()
        """

        return x, [n_H, n_W]

    def grids_inverse(self, x, n_crops, level):
        crop_size = self.crop_size // (2**level)
        stride_crop = self.stride_crop // (2**level)
        
        n_H, n_W = n_crops
        B_total, C, H, W = x.shape

        #out_H = H + 2 * stride_crop
        #out_W = W + 2 * stride_crop
        out_H = (n_H - 1)* stride_crop + crop_size
        out_W = (n_W - 1)* stride_crop + crop_size

        B = B_total//(n_H*n_W)
        x = Rearrange('(b oh ow) c ph pw -> b (c ph pw) (oh ow)', c=C, ph=crop_size, pw=crop_size, oh=n_H, ow=n_W)(x)

        counts = torch.ones_like(x)
        
        x = F.fold(x, output_size=(out_H, out_W) , kernel_size=(crop_size, crop_size), padding=(0, 0), stride=(stride_crop, stride_crop))
        counts = F.fold(counts, output_size=(out_H, out_W) , kernel_size=(crop_size, crop_size), padding=(0, 0), stride=(stride_crop, stride_crop))

        x = x / counts
        
        return x


    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        inp_identity = inp
        inp, n_crops = self.grids(inp)

        x = self.intro(inp)

        self.b_encoder.eval()

        embedding = self.b_encoder(inp)
        embedding = embedding / embedding.norm(dim=1, keepdim=True)

        x_hyper = F.gelu(self.norm2(self.fc_hyper1(embedding)))
        #x_hyper = self.mlp_res_block1(x_hyper)
        #x_hyper = self.mlp_res_block2(x_hyper)
        x_hyper = F.gelu(self.norm3(self.fc_hyper2(x_hyper)))
        x_hyper = self.fc_hyper5(x_hyper)
        """
        x_hyper = self.mlp_res_block1(x_hyper)
        x_hyper = self.mlp_res_block2(x_hyper)
        
        x_hyper_0 = F.gelu(self.fc_hyper2_0(x_hyper))
        x_hyper_1 = F.gelu(self.fc_hyper2_1(x_hyper))
        x_hyper_2 = F.gelu(self.fc_hyper2_2(x_hyper))

        x_hyper_0 = self.fc_hyper3_0(x_hyper_0)
        x_hyper_1 = self.fc_hyper3_1(x_hyper_1)
        x_hyper_2 = self.fc_hyper3_2(x_hyper_2)
        
        x_hyper = torch.cat([x_hyper_0, x_hyper_1, x_hyper_2], dim=-1)
        """

        weights_and_biases = self.parse_weights_and_biases(x_hyper)

        encs = []
        for i, down in enumerate(self.downs):
            #print(i, x.shape)
            if i<self.n_hyper:
                x = self.encoders_hyper[i](x, weights_and_biases[i])
                x_shortcut = self.grids_inverse(x, n_crops, i)

                encs.append(x_shortcut)
                
                if i == self.n_hyper-1:
                    x = x_shortcut
                x = down(x)    
                #print(i, x_shortcut.shape)
            else:
                x = self.encoders[i-self.n_hyper](x)
                encs.append(x)
                x = down(x)    
            
        """
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        """
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        """
        #for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
        encs_reverse = encs[::-1]
        count=0
        for i, up in enumerate(self.ups):
            x = up(x)
            x = x + encs_reverse[i]
            if i<self.n_hyper:
                #print(f'default {i}')
                x = self.decoders[i](x)
            else:
                #print(f'hyper {i-self.n_hyper}')
                
                x = self.decoders_hyper[i-self.n_hyper](x, weights_and_biases[self.n_hyper+count])
                count+=1
        """
        x = self.ending(x)

        x = x + inp_identity

        return x[:, :, :H, :W]
    
    def parse_weights_and_biases(self, x):
        weights_and_biases = [{} for _ in range(self.n_hyper)]
        start = 0
        for k, v in self.conv_params_dict.items():
            _, i_block, i_layer, name_conv, name_param = k.split('.')
            n = v['n_elements']
            #in_plane, out_plane, kH, kW = v['shape']
            end = start + n
            if name_conv not in weights_and_biases[int(i_block)]:
                weights_and_biases[int(i_block)][name_conv] = {'weight': None, 'bias': None}
            weights_and_biases[int(i_block)][name_conv][name_param] = x[..., start:end].reshape(v['shape'])
            
            start=end
        return weights_and_biases


    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetBlurCLIPLocal(Local_Base, NAFNetBlurCLIP):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNetBlurCLIP.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        #self.eval()
        #with torch.no_grad():
        #    self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
