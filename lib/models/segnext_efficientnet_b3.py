import json
import math

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm

"""
    [batch_size, in_channels, height, width] -> [batch_size, out_channels, height // 4, width // 4]
"""



class LightHamHead(nn.Module):

    def __init__(
            self,
            in_channels_list=[64, 160, 256],
            hidden_channels=256,
            out_channels=256,
            classes_num=150,
            drop_prob=0.1,
    ):
        super(LightHamHead, self).__init__()

        self.cls_seg = nn.Sequential(
            nn.Dropout2d(drop_prob),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=classes_num,
                kernel_size=(1, 1)
            )
        )

        self.squeeze = nn.Sequential(
            nn.Conv2d(
                in_channels=sum(in_channels_list),
                out_channels=hidden_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=hidden_channels,
            ),
            nn.ReLU()
        )

        self.align = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=out_channels
            ),
            nn.ReLU()
        )

    # inputs: [x, x_1, x_2, x_3]
    # x: [batch_size, channels, height, width]
    def forward(self, inputs):
        assert len(inputs) >= 2
        o = inputs[0]
        batch_size, _, standard_height, standard_width = inputs[1].shape
        standard_shape = (standard_height, standard_width)
        inputs = [
            F.interpolate(
                input=x,
                size=standard_shape,
                mode="bilinear",
                align_corners=False
            )
            for x in inputs[1:]
        ]

        # x: [batch_size, channels_1 + channels_2 + channels_3, standard_height, standard_width]
        x = torch.cat(inputs, dim=1)

        # out: [batch_size, channels_1 + channels_2 + channels_3, standard_height, standard_width]
        out = self.squeeze(x)
        out = self.align(out)

        # out: [batch_size, classes_num, standard_height, standard_width]
        out = self.cls_seg(out)

        _, _, original_height, original_width = o.shape
        # out: [batch_size, original_height * original_width, classes_num]
        out = F.interpolate(
            input=out,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False
        )
        # out = torch.transpose(out.view(batch_size, -1, original_height * original_width), -2, -1)

        return out


class SegNeXt_EfficientNet_B3(nn.Module):

    def __init__(
            self,
            embed_dims=[3, 32, 48, 136, 384],
            hidden_channels=256,
            out_channels=256,
            n_classes=150,
            aux_mode='train',
            drop_prob_of_decoder=0.1,
            use_fp16=False,
    ):
        super(SegNeXt_EfficientNet_B3, self).__init__()

        self.aux_mode=aux_mode
        self.use_fp16 = use_fp16
        self.out_indices = [2,3,4]
        self.encoder = timm.create_model('tf_efficientnet_b3.ns_jft_in1k', features_only=True, out_indices=self.out_indices,pretrained=True)

        self.decoder = LightHamHead(
            in_channels_list=embed_dims[-3:],
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            classes_num=n_classes,
            drop_prob=drop_prob_of_decoder,
        )

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            out = self.encoder(x)
            out.insert(0,x)
            out = self.decoder(out)
            if self.aux_mode == 'train':
                return out
            elif self.aux_mode == 'eval':
                return out,
            else:
                out = torch.argmax(out, dim=1)
                out = torch.tensor(out, dtype=torch.float32)
                return out


if __name__ == "__main__":
    net = SegNeXt_EfficientNet_B3(n_classes=9)
    net.eval()
    in_ten = torch.randn(2, 3, 224, 224)
    aa = net(in_ten)
    print(aa.shape)