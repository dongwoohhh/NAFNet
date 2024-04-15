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

    def normalize_weights(self, w):
        norm_w = Rearrange('b co ci kh kw -> b co ci (kh kw) 1')(w)
        norm_w = torch.norm(norm_w, dim=(2, 3), keepdim=True)
        
        w = w / norm_w

        return w

    def forward(self, inp, weights_and_bias):
        x = inp
        B = x.shape[0]

        x = self.norm1(x)
        
        w1 = self.normalize_weights(weights_and_bias['conv1']['weight'])
        b1 = weights_and_bias['conv1']['bias']
        #import pdb; pdb.set_trace()
        #x = [F.conv2d(x[i:i+1], self.w1+w1[i], self.b1+b1[i], padding=0) for i in range(B)]
        x = [F.conv2d(x[i:i+1], w1[i], b1[i] if b1 is not None else None, padding=0) for i in range(B)]
        x = torch.cat(x, dim=0)
        
        w2 = self.normalize_weights(weights_and_bias['conv2']['weight'])
        b2 = weights_and_bias['conv2']['bias']
        #x = [F.conv2d(x[i:i+1], self.w2+w2[i], self.b2+b2[i], padding=1, groups=self.dw_channel) for i in range(B)]
        x = [F.conv2d(x[i:i+1], w2[i], b2[i] if b1 is not None else None, padding=1, groups=self.dw_channel) for i in range(B)]
        x = torch.cat(x, dim=0)
        
        x = self.sg(x)
        x = x * self.sca(x)
        
        w3 = self.normalize_weights(weights_and_bias['conv3']['weight'])
        b3 = weights_and_bias['conv3']['bias']
        #x = [F.conv2d(x[i:i+1], self.w3+w3[i], self.b3+b3[i], padding=0) for i in range(B)]
        x = [F.conv2d(x[i:i+1], w3[i], b3[i] if b1 is not None else None, padding=0) for i in range(B)]
        x = torch.cat(x, dim=0)
        

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)

        w4 = self.normalize_weights(weights_and_bias['conv4']['weight'])
        b4 = weights_and_bias['conv4']['bias']
        #x = [F.conv2d(x[i:i+1], self.w4+w4[i], self.b4+b4[i], padding=0) for i in range(B)]
        x = [F.conv2d(x[i:i+1], w4[i], b4[i] if b1 is not None else None, padding=0) for i in range(B)]
        x = torch.cat(x, dim=0)

        x = self.sg(x)

        w5 = self.normalize_weights(weights_and_bias['conv5']['weight'])
        b5 = weights_and_bias['conv5']['bias']
        #x = [F.conv2d(x[i:i+1], self.w5+w5[i], self.b5+b5[i], padding=0) for i in range(B)]
        x = [F.conv2d(x[i:i+1], w5[i], b5[i] if b1 is not None else None, padding=0) for i in range(B)]
        x = torch.cat(x, dim=0)

        x = self.dropout2(x)
        
        return y + x * self.gamma



class NAFBlockModulated(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()

        dw_channel = c * DW_Expand
        self.dw_channel = dw_channel
        self.c = c
        #self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        #self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        #self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        
        self.w1 = nn.Parameter(torch.randn(dw_channel, c, 1, 1), requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(dw_channel), requires_grad=True)
        nn.init.kaiming_uniform_(self.w1, mode='fan_in', nonlinearity='relu')
        
        self.w2 = nn.Parameter(torch.randn(dw_channel, 1, 3, 3), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(dw_channel))
        nn.init.kaiming_uniform_(self.w2, mode='fan_in', nonlinearity='relu')
        

        self.w3 = nn.Parameter(torch.randn(c, dw_channel//2, 1, 1), requires_grad=True)
        self.b3 = nn.Parameter(torch.zeros(c))
        nn.init.kaiming_uniform_(self.w3, mode='fan_in', nonlinearity='relu')
        
        
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

        self.w4 = nn.Parameter(torch.randn(ffn_channel, c, 1, 1), requires_grad=True)
        self.b4 = nn.Parameter(torch.zeros(ffn_channel), requires_grad=True)
        nn.init.kaiming_uniform_(self.w4, mode='fan_in', nonlinearity='relu')
        
        self.w5 = nn.Parameter(torch.randn(c, ffn_channel//2, 1, 1), requires_grad=True)
        self.b5 = nn.Parameter(torch.zeros(c), requires_grad=True)
        nn.init.kaiming_uniform_(self.w5, mode='fan_in', nonlinearity='relu')


        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


    def normalize_weights(self, w):
        norm_w = Rearrange('b co ci kh kw wh ww-> b co ci (kh kw) 1 wh ww')(w)

        norm_w = torch.norm(norm_w, dim=(2, 3), keepdim=True)
        
        w = w / norm_w

        return w


    def conv1x1_pixelwise(self, x, weights_and_biases):
        w = weights_and_biases['weight']
        b = weights_and_biases['bias']

        w = w.squeeze((3,4))
        #identity = x

        B, _, H, W = x.shape
        _, Co, Ci, wH, wW = w.shape
        
        w = Rearrange('b co ci h w -> b (co ci) h w')(w)
        
        
        w = F.interpolate(w, size=(H, W), mode='bilinear')
        w = w.reshape(B, Co, Ci, H, W)
        w = Rearrange('b co ci h w -> b h w co ci')(w)#.reshape(B*H*W, Co, Ci)

        b = F.interpolate(b, size=(H, W), mode='bilinear')
        b = Rearrange('b co h w -> b h w co')(b)

        x = Rearrange('b c h w -> b h w c')(x).unsqueeze(-1)#.reshape(B*H*W, Ci)

        x = torch.matmul(w, x)
        x = x.squeeze(-1)
        x = x + b
        x = Rearrange('b h w c -> b c h w')(x)

        #x = x+identity

        return x
    
    def conv1x1_group(self, x, weights_and_biases):
        w = weights_and_biases['weight']
        b = weights_and_biases['bias']

        w = w.squeeze((3,4))
        #identity = x

        B, _, H, W = x.shape
        _, Co, Ci, wH, wW = w.shape
        
        patch_H = H // wH
        patch_W = W // wW


        w = Rearrange('b co ci h w -> b ci h w co')(w)
        w = w.reshape(B, Ci, wH*wW*Co)
        w = Rearrange('b ci ca -> b ca ci')(w)
        w = Rearrange('b ca ci -> (b ca) ci')(w)[..., None, None]

        b = Rearrange('b co h w -> b h w co')(b)
        b = b.reshape(B*wH*wW*Co)


        x = F.unfold(x, kernel_size=(patch_H, patch_W), padding=(0, 0), stride=(patch_H, patch_W))
        x = x.reshape(B, Ci, patch_H, patch_W, wH, wW)
        x = Rearrange('b ci ph pw wh ww -> b wh ww ci ph pw')(x)
        x = x.reshape(B*wH*wW*Ci, patch_H, patch_W)
        """
        x = x.reshape(B, Ci, wH, patch_H, wW, patch_W)
        x = Rearrange('b ci wh ph ww pw -> b wh ww ci ph pw')(x)
        x = x.reshape(B*wH*wW*Ci, patch_H, patch_W)
        """
        #x_list = [F.conv2d(x[i:i+1], w[i], b[i],stride=1, padding=0, dilation=1, groups=wH*wW) for i in range(B)]
        #x = torch.cat(x_list, dim=0)
        x = F.conv2d(x, w, b, stride=1, padding=0, dilation=1, groups=B*wH*wW)

        x = x.reshape(B, wH, wW, Co, patch_H, patch_W)
        x = Rearrange('b wh ww co ph pw -> b co wh ph ww pw')(x)
        x = x.reshape(B, Co, H, W)

        return x
    
    def conv3x3_group(self, x, weights_and_biases):
        w = weights_and_biases['weight']
        b = weights_and_biases['bias']

        #w = w.squeeze((3,4))
        #identity = x
        
        B, _, H, W = x.shape
        _, Co, Ci, kH, kW, wH, wW = w.shape
        
        patch_H = H // wH
        patch_W = W // wW


        w = Rearrange('b co ci kh kw h w -> b ci h w co kh kw')(w)
        w = w.reshape(B, Ci, wH*wW*Co, kH, kW)
        w = Rearrange('b ci ca kh kw -> b ca ci kh kw')(w)
        w = Rearrange('b ca ci kh kw -> (b ca) ci kh kw')(w)
        #w = w[..., None, None]

        b = Rearrange('b co h w -> b h w co')(b)
        b = b.reshape(B*wH*wW*Co)

        
        x = F.unfold(x, kernel_size=(patch_H+2, patch_W+2), padding=(1, 1), stride=(patch_H, patch_W))
        x = x.reshape(B, Co, patch_H+2, patch_W+2, wH, wW)
        x = Rearrange('b co ph pw wh ww -> b wh ww co ph pw')(x)
        x = x.reshape(B*wH*wW*Co, patch_H+2, patch_W+2)
        x = F.conv2d(x, w, b, stride=1, padding=0, dilation=1, groups=B*wH*wW*Co)
        #x1_tmp = x1[:, :, 1:-1, 1:-1]
        #x1_tmp = Rearrange('a b c d e f -> a b e c f d')(x1_tmp)
        #x1_tmp = x1_tmp.reshape(B, Co, H, W)
        """
        x = x.reshape(B, Co, wH, patch_H, wW, patch_W)
        x = Rearrange('b co wh ph ww pw -> b wh ww co ph pw')(x)
        x = x.reshape(B*wH*wW*Co, patch_H, patch_W)

        #x_list = [F.conv2d(x[i:i+1], w[i], b[i], stride=1, padding=1, dilation=1, groups=wH*wW*Co) for i in range(B)]
        #x = torch.cat(x_list, dim=0)
        x = F.conv2d(x, w, b, stride=1, padding=1, dilation=1, groups=B*wH*wW*Co)
        """
        x = x.reshape(B, wH, wW, Co, patch_H, patch_W)
        x = Rearrange('b wh ww co ph pw -> b co wh ph ww pw')(x)
        x = x.reshape(B, Co, H, W)

        return x
    
    def conv1x1_modulated(self, x, w_base, b_base, weights_and_biases):
        modulation = weights_and_biases['weight']
        #b = weights_and_biases['bias']
        
        #w = w.squeeze((3,4))
        #identity = x
        
        w = w_base[None, ..., None, None] * modulation
        b = b_base[None, ..., None, None]

        w = self.normalize_weights(w)

        B, _, H, W = x.shape
        _, Co, Ci, kH, kW, wH, wW = w.shape
        
        patch_H = H // wH
        patch_W = W // wW

        w = Rearrange('b co ci kh kw h w -> b ci h w co kh kw')(w)
        w = w.reshape(B, Ci, wH*wW*Co, kH, kW)
        w = Rearrange('b ci ca kh kw -> b ca ci kh kw')(w)
        w = Rearrange('b ca ci kh kw -> (b ca) ci kh kw')(w)


        #w = Rearrange('b co ci h w -> b ci h w co')(w)
        #w = w.reshape(B, Ci, wH*wW*Co)
        #w = Rearrange('b ci ca -> b ca ci')(w)
        #w = Rearrange('b ca ci -> (b ca) ci')(w)[..., None, None]

        #b = Rearrange('b co h w -> b h w co')(b)
        #b = b.reshape(B*wH*wW*Co)


        x = F.unfold(x, kernel_size=(patch_H, patch_W), padding=(0, 0), stride=(patch_H, patch_W))
        x = x.reshape(B, Ci, patch_H, patch_W, wH, wW)
        x = Rearrange('b ci ph pw wh ww -> b wh ww ci ph pw')(x)
        x = x.reshape(B*wH*wW*Ci, patch_H, patch_W)

        """
        x = x.reshape(B, Ci, wH, patch_H, wW, patch_W)
        x = Rearrange('b ci wh ph ww pw -> b wh ww ci ph pw')(x)
        x = x.reshape(B*wH*wW*Ci, patch_H, patch_W)
        """
        #x_list = [F.conv2d(x[i:i+1], w[i], b[i],stride=1, padding=0, dilation=1, groups=wH*wW) for i in range(B)]
        #x = torch.cat(x_list, dim=0)
        x = F.conv2d(x, w, None, stride=1, padding=0, dilation=1, groups=B*wH*wW)

        x = x.reshape(B, wH, wW, Co, patch_H, patch_W)
        x = Rearrange('b wh ww co ph pw -> b co wh ph ww pw')(x)
        x = x.reshape(B, Co, H, W)

        x = x + b

        return x
    
    def conv3x3_modulated(self, x, w_base, b_base, weights_and_biases):
        modulation = weights_and_biases['weight']

        #w = w.squeeze((3,4))
        #identity = x
        
        w = w_base[None, ..., None, None] * modulation
        b = b_base[None, ..., None, None]

        w = self.normalize_weights(w)

        B, _, H, W = x.shape
        _, Co, Ci, kH, kW, wH, wW = w.shape
        
        patch_H = H // wH
        patch_W = W // wW


        w = Rearrange('b co ci kh kw h w -> b ci h w co kh kw')(w)
        w = w.reshape(B, Ci, wH*wW*Co, kH, kW)
        w = Rearrange('b ci ca kh kw -> b ca ci kh kw')(w)
        w = Rearrange('b ca ci kh kw -> (b ca) ci kh kw')(w)
        #w = w[..., None, None]

        #b = Rearrange('b co h w -> b h w co')(b)
        #b = b.reshape(B*wH*wW*Co)

        
        x = F.unfold(x, kernel_size=(patch_H+2, patch_W+2), padding=(1, 1), stride=(patch_H, patch_W))
        x = x.reshape(B, Co, patch_H+2, patch_W+2, wH, wW)
        x = Rearrange('b co ph pw wh ww -> b wh ww co ph pw')(x)
        x = x.reshape(B*wH*wW*Co, patch_H+2, patch_W+2)
        x = F.conv2d(x, w, None, stride=1, padding=0, dilation=1, groups=B*wH*wW*Co)
        #x1_tmp = x1[:, :, 1:-1, 1:-1]
        #x1_tmp = Rearrange('a b c d e f -> a b e c f d')(x1_tmp)
        #x1_tmp = x1_tmp.reshape(B, Co, H, W)
        """
        x = x.reshape(B, Co, wH, patch_H, wW, patch_W)
        x = Rearrange('b co wh ph ww pw -> b wh ww co ph pw')(x)
        x = x.reshape(B*wH*wW*Co, patch_H, patch_W)

        #x_list = [F.conv2d(x[i:i+1], w[i], b[i], stride=1, padding=1, dilation=1, groups=wH*wW*Co) for i in range(B)]
        #x = torch.cat(x_list, dim=0)
        x = F.conv2d(x, w, b, stride=1, padding=1, dilation=1, groups=B*wH*wW*Co)
        """
        x = x.reshape(B, wH, wW, Co, patch_H, patch_W)
        x = Rearrange('b wh ww co ph pw -> b co wh ph ww pw')(x)
        x = x.reshape(B, Co, H, W)

        x = x + b

        return x
    
    def forward(self, inp, weights_and_bias):
        x = inp
        B = x.shape[0]

        x = self.norm1(x)

        #x = self.conv1x1_group(x, weights_and_bias['conv1'])
        x = self.conv1x1_modulated(x, self.w1, self.b1, weights_and_bias['conv1'])
        

        """
        w2 = weights_and_bias['conv2']['weight']
        b2 = weights_and_bias['conv2']['bias']
        w2 = w2.squeeze(2)

        B, C, kH, kW, wH, wW = w2.shape
        _, _, H, W = x.shape

        w2 = Rearrange('b c kh kw h w -> (b c) (kh kw) h w')(w2)
        
        w2 = F.interpolate(w2, size=(H, W), mode='bilinear')
        w2 = w2.reshape(B, C, kH*kW, H, W)
        w2 = w2.contiguous()

        b2 = F.interpolate(b2, size=(H, W), mode='bilinear')
        
        x = x.contiguous()
        x = [ddf(x[i].unsqueeze(1), torch.ones((C, 1, kH, kW) , device=torch.device('cuda')), w2[i], 3, 1, 1, 'mul').squeeze(1) for i in range(B)] #, 'f'
        x = torch.stack(x, dim=0)

        x = x + b2
        """
        #x = self.conv3x3_group(x, weights_and_bias['conv2'])
        x = self.conv3x3_modulated(x, self.w2, self.b2, weights_and_bias['conv2'])

        x = self.sg(x)
        x = x * self.sca(x)
        
        #x = self.conv1x1_group(x, weights_and_bias['conv3'])
        x = self.conv1x1_modulated(x, self.w3, self.b3, weights_and_bias['conv3'])

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        
        #x = self.conv1x1_group(x, weights_and_bias['conv4'])
        x = self.conv1x1_modulated(x, self.w4, self.b4, weights_and_bias['conv4'])
        
        x = self.sg(x)
        
        #x = self.conv1x1_group(x, weights_and_bias['conv5'])
        x = self.conv1x1_modulated(x, self.w5, self.b5, weights_and_bias['conv5'])

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

    def __init__(self, pretrained_clip_dir, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], vision_layers=[3,4,6,3], embed_dim=128):
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
        self.b_encoder = BlurEncoder(layers=vision_layers, output_dim=embed_dim, width=64)
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
        chan_middle = chan
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
        """
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
        """
        n_params= 0
        n_params0 = 0
        n_params1 = 0
        n_params2 = 0
        n_params3 = 0  
        #name.startswith('decoders.0') or
        #or name.endswith('bias'))
        for name, param in self.named_parameters(): 
            if (name.startswith('decoders.0') or name.startswith('decoders.1') or name.startswith('decoders.2') or name.startswith('decoders.3')) and \
                (name.endswith('weight')) and \
                    name.find('conv')>0 and name.find('b_encoder')<0:
                dims=[-1]
                #dims.extend(list(param.shape))
                dims.extend(list([1, param.shape[1],1,1]))
                #print(name, param.shape)
                #self.conv_params_dict[name] = {'n_elements': param.nelement(), 'shape':dims}
                self.conv_params_dict[name] = {'n_elements': param.shape[1], 'shape':dims}
                #print(self.conv_params_dict[name])
                #n_params = n_params+param.nelement()
                n_params = n_params+param.shape[0]
                
                if name.startswith('decoders.0'):
                    #n_params0 += param.nelement()
                    n_params0 = n_params0+param.shape[1]
                if name.startswith('decoders.1'):
                    #n_params1 += param.nelement()
                    n_params1 = n_params1+param.shape[1]
                if name.startswith('decoders.2'):
                    #n_params2 += param.nelement()
                    n_params2 = n_params2+param.shape[1]
                if name.startswith('decoders.3'):
                    #n_params3 += param.nelement()
                    n_params3 = n_params3+param.shape[1]

        print(f"##### DECODER PARAMS #{str(n_params0)}, {str(n_params1)}, {str(n_params2)}, {str(n_params3)} {str(n_params)}")

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
        """
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
        self.n_hyper = 3
        chan = chan_middle

        self.decoders_hyper = nn.ModuleList()
        count = 0
        for i, num in enumerate(dec_blk_nums):
            chan = chan //2
            if i < len(dec_blk_nums) - self.n_hyper:
                continue
            else:
                assert num == 1
                self.decoders_hyper.append(
                    NAFBlockModulated(chan)
                )
                self.decoders.pop(i-count)
                count=count+1

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

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        self.b_encoder.eval()
        embedding = self.b_encoder(inp)
        embedding = embedding / embedding.norm(dim=1, keepdim=True)

        embedding = Rearrange('b c h w -> b h w c')(embedding)
        
        x_hyper = F.gelu(self.norm2(self.fc_hyper1(embedding)))
        x_hyper = F.gelu(self.norm3(self.fc_hyper2(x_hyper)))
        x_hyper = self.fc_hyper5(x_hyper)

        x_hyper = Rearrange('b h w c -> b c h w')(x_hyper)

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
        """
        for i, down in enumerate(self.downs):
            if i<self.n_hyper:
                x = self.encoders_hyper[i](x, weights_and_biases[i])
            else:
                x = self.encoders[i-self.n_hyper](x)
            encs.append(x)
            x = down(x)    
        """
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        x = self.middle_blks(x)
        """
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        """
        #for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
        encs_reverse = encs[::-1]
        count=0
        n_vanilla = len(self.ups)-self.n_hyper
        for i, up in enumerate(self.ups):
            x = up(x)
            x = x + encs_reverse[i]
            if i<n_vanilla:
                #print(f'default {i}')
                x = self.decoders[i](x)
            else:
                #print(f'hyper {i-self.n_hyper}')
                x = self.decoders_hyper[count](x, weights_and_biases[n_vanilla+count])
                count+=1
        
        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]
    
    def parse_weights_and_biases(self, x):
        B, _, H, W = x.shape
        weights_and_biases = [{} for _ in range(len(self.encoders))]
        
        start = 0
        for k, v in self.conv_params_dict.items():
            _, i_block, i_layer, name_conv, name_param = k.split('.')
            n = v['n_elements']
            #in_plane, out_plane, kH, kW = v['shape']
            end = start + n
            if name_conv not in weights_and_biases[int(i_block)]:
                weights_and_biases[int(i_block)][name_conv] = {'weight': None, 'bias': None}
            shape = v['shape'] + [H, W]

            #weights_and_biases[int(i_block)][name_conv][name_param] = x[..., start:end].reshape(v['shape'])
            weights_and_biases[int(i_block)][name_conv][name_param] = x[:, start:end].reshape(shape)
            
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

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


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
