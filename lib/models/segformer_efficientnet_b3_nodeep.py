from math import sqrt
from functools import partial
import torch
from torch import nn, einsum

from torch.cuda.amp import autocast
from timm.layers.squeeze_excite import EffectiveSEModule
import timm

# helpers
def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


# classes
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = torch.nn.BatchNorm2d(out_chan)
        self.relu = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = nn.Conv2d(in_chan, mid_chan,kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.Upsample(scale_factor=up_factor,
                mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.se=EffectiveSEModule(out_chan)

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        feat_atten =self.se(feat)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Segformer_EfficientNet_B3_NODeep(nn.Module):
    def __init__(self,
        dims = (32, 48, 136, 384),
        decoder_dim = 256,
        n_classes=9,
        aux_mode='train',
        use_fp16=False
    ):
        super(Segformer_EfficientNet_B3_NODeep,self).__init__()
        dims=tuple(dims)

        self.out_indices = [1,2,3,4]
        self.selected_feature_extractor = timm.create_model('tf_efficientnet_b3.ns_jft_in1k', features_only=True, out_indices=self.out_indices,pretrained=True)
        self.use_fp16 = use_fp16
        self.aux_mode=aux_mode
        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        self.se=EffectiveSEModule( decoder_dim*4)
        self.conv_out = BiSeNetOutput(decoder_dim*4, decoder_dim, n_classes, up_factor=4)

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            layer_outputs = self.selected_feature_extractor(x)
            fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
            fused = torch.cat(fused, dim = 1)
            fused=self.se(fused)
            feat_out=self.conv_out(fused)
            if self.aux_mode == 'train':
                return feat_out
            elif self.aux_mode == 'eval':
                return feat_out,
            elif self.aux_mode == 'pred':
                feat_out=torch.argmax(feat_out,dim=1)
                feat_out=torch.tensor(feat_out,dtype=torch.float32)
                return feat_out


if __name__ == "__main__":
    net = Segformer_EfficientNet_B3_NODeep(n_classes=9)
    net.eval()
    in_ten = torch.randn(8, 3,224, 224)
    out= net(in_ten)
    print(out.shape)
