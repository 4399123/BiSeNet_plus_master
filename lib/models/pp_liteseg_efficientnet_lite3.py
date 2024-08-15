#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np
from .efficientnet_lite3 import EfficientNet_Lite3

from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.Hardswish(inplace=True)
        # self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class SPPFModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 4  # hidden channels
        self.cv1 = ConvBNReLU(in_chan=in_channels, out_chan=c_, ks=1, stride=1, padding=0)
        self.cv2 = ConvBNReLU(in_chan=c_ * 4, out_chan=out_channels, ks=1, stride=1, padding=0)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class UAFM(nn.Module):
    """Unified Attention Fusion Modul
    """

    def __init__(self, low_chan, hight_chan, out_chan, u_type='sp') -> None:
        """
        :param low_chan: int, channels of input low-level feature
        :param hight_chan: int, channels of input high-level feature
        :param out_chan: int, channels of output faeture
        :param u_type: string, attention type, sp: spatial attention, ch: channel attention
        """
        super().__init__()
        self.u_type = u_type

        if u_type == 'sp':
            self.conv_atten = nn.Sequential(
                ConvBNReLU(in_chan=4, out_chan=2, kernel_size=3),
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3,3), padding=(1,1), bias=False),
                nn.BatchNorm2d(1),
            )
        else:
            self.conv_atten = nn.Sequential(
                ConvBNReLU(in_chan=4 * hight_chan, out_chan=hight_chan // 2, kernel_size=1, padding=0),
                nn.Conv2d(hight_chan // 2, hight_chan, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(hight_chan),
            )

        self.conv_low = ConvBNReLU(in_chan=low_chan, out_chan=hight_chan, kernel_size=3, padding=1, bias=False)
        self.conv_out = ConvBNReLU(in_chan=hight_chan, out_chan=out_chan, kernel_size=3, padding=1, bias=False)

    def _spatial_attention(self, x):
        """
        :param x: tensor, feature
        :return x: tensor, fused feature
        """
        mean_value = torch.max(x, dim=1, keepdim=True)[0]
        max_value = torch.mean(x, dim=1, keepdim=True)

        value = torch.concat([mean_value, max_value], dim=1)

        return value

    def _channel_attention(self, x):
        """
        :param x: tensor, feature
        :return x: tensor, fused feature
        """
        H,W=x.size()[2:]
        inputsz=np.array([H,W])
        outputsz = np.array([1, 1])
        stridesz = np.floor(inputsz / outputsz).astype(np.int32)
        kernelsz = inputsz - (outputsz - 1) * stridesz
        avg_value=F.avg_pool2d(x,kernel_size=list(kernelsz),stride=list(stridesz))
        # avg_value = F.adaptive_avg_pool2d(x, 1)
        max_value = torch.max(torch.max(x, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        value = torch.concat([avg_value, max_value], dim=1)

        return value

    def forward(self, x_high, x_low):
        """
        :param x_high: tensor, high-level feature
        :param x_low: tensor, low-level feature
        :return x: tensor, fused feature
        """
        h, w = x_low.size()[2:]

        x_low = self.conv_low(x_low)
        x_high = F.interpolate(x_high, (h, w), mode='bilinear', align_corners=False)

        if self.u_type == 'sp':
            atten_high = self._spatial_attention(x_high)
            atten_low = self._spatial_attention(x_low)
        else:
            atten_high = self._channel_attention(x_high)
            atten_low = self._channel_attention(x_low)

        atten = torch.concat([atten_high, atten_low], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))

        x = x_high * atten + x_low * (1 - atten)
        x = self.conv_out(x)

        return x


class SegHead(nn.Module):
    """FLD Decoder
    """

    def __init__(self, decode_chans):
        """
        :param bin_sizes: list, avg pool size of 3 features
        :param decode_chans: list, channels of decoder feature size
        """
        super().__init__()

        self.sppm = SPPFModule(384, decode_chans[0])
        self.uafm1 = UAFM(136, decode_chans[0], decode_chans[1])
        self.uafm2 = UAFM(48, decode_chans[1], decode_chans[2])

    def forward(self, x):
        # x8, x16, x32
        sppm_feat = self.sppm(x[-1])

        merge_feat1 = self.uafm1(sppm_feat, x[1])
        merge_feat2 = self.uafm2(merge_feat1, x[0])

        # return [sppm_feat, merge_feat1, merge_feat2]
        return [merge_feat2, merge_feat1, sppm_feat]


class SegClassifier(nn.Module):
    """Classification Layer
    """

    def __init__(self, in_chan, mid_chan, n_classes) -> None:
        """
        :param in_chan: int, channels of input feature
        :param mid_chan: int, channels of mid conv
        :param n_classes: int, number of classification
        """
        super().__init__()
        self.conv = ConvBNReLU(in_chan=in_chan, out_chan=mid_chan, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)

        return x


class PPLiteSeg_EfficientNet_Lite3(nn.Module):

    def __init__(self, n_classes, aux_mode='train',use_fp16=False, *args, **kwargs):
        super(PPLiteSeg_EfficientNet_Lite3,self).__init__()

        self.aux_mode = aux_mode
        self.use_fp16 = use_fp16
        decode_chans = [128, 96, 64]

        self.resnet = EfficientNet_Lite3()
        self.seg_head = SegHead(decode_chans)

        self.classifer = []
        for chan in decode_chans[::-1]:
            cls = SegClassifier(chan, 64, n_classes)
            self.classifer.append(cls)
        self.classifer = nn.Sequential(*self.classifer)

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            h, w = x.size()[2:]
            outs=[]
            feats_selected=[None,None,None]
            out8,out16,out32=self.resnet(x)
            feats_selected[0],feats_selected[1],feats_selected[2]=out8,out16,out32
            head_out = self.seg_head(feats_selected)
            out_main=self.classifer[0](head_out[0])
            feat_out=F.interpolate(out_main, (h, w), mode='bilinear', align_corners=False)
            outs.append(feat_out)
            if self.aux_mode == 'train':
                for i in range(1,3):
                    out_aux = self.classifer[i](head_out[i])
                    out_aux= F.interpolate(out_aux, (h, w), mode='bilinear', align_corners=False)
                    outs.append(out_aux)
                return outs[0],outs[1],outs[2]
            elif self.aux_mode == 'eval':
                return outs[0],
            elif self.aux_mode == 'pred':
                feat_out=torch.argmax(feat_out,dim=1)
                feat_out=torch.tensor(feat_out,dtype=torch.float32)
                return feat_out

if __name__ == "__main__":
    net = PPLiteSeg_EfficientNet_Lite3(19)
    net.eval()
    in_ten = torch.randn(2, 3,224, 224)
    out, out16, out32 = net(in_ten)
    print(out.shape)
    print(out16.shape)
    print(out32.shape)

