# following from: https://github.com/Cadene/pretrained-models.pytorch
# LICENSE of Original
"""
BSD 3-Clause License

Copyright (c) 2017, Remi Cadene
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
# Modified by Gavin Gray for Mobile, CIFAR and alternative Cell structures

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

pretrained_settings = {
    'nasnetalarge': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth',
            'input_space': 'RGB',
            'input_size': [3, 331, 331], # resize 354
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth',
            'input_space': 'RGB',
            'input_size': [3, 331, 331], # resize 354
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}

class MaxPool(nn.Module):

    def __init__(self, pad=False):
        super(MaxPool, self).__init__()
        self.pad = pad
        self.pad = nn.ZeroPad2d((1, 0, 1, 0)) if pad else None
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        if self.pad:
            x = self.pad(x)
        x = self.pool(x)
        if self.pad:
            x = x[:, :, 1:, 1:]
        return x


class AvgPool(nn.Module):

    def __init__(self, pad=False, stride=2, padding=1):
        super(AvgPool, self).__init__()
        self.pad = pad
        self.pad = nn.ZeroPad2d((1, 0, 1, 0)) if pad else None
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding)

    def forward(self, x):
        if self.pad:
            x = self.pad(x)
        x = self.pool(x)
        if self.pad:
            x = x[:, :, 1:, 1:]
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, reduction=False, z_padding=1):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        if reduction:
            self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x) if hasattr(self, 'padding') else x
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:].contiguous() if hasattr(self, 'padding') else x
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x



class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):

    def __init__(self, num_conv_filters, stem_multiplier):
        super(CellStem0, self).__init__()
        nf1, nf2 = 32*stem_multiplier, num_conv_filters//4
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(nf1, nf2, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(nf2, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(nf2, nf2, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(nf1, nf2, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(nf1, nf2, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(nf1, nf2, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(nf2, nf2, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self, num_conv_filters, stem_multiplier):
        super(CellStem1, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(num_conv_filters, num_conv_filters//2, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(num_conv_filters//2, eps=0.001, momentum=0.1, affine=True))

        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(32*stem_multiplier, num_conv_filters//4, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(32*stem_multiplier, num_conv_filters//4, 1, stride=1, bias=False))

        nf = num_conv_filters//2
        self.final_path_bn = nn.BatchNorm2d(nf, eps=0.001, momentum=0.1, affine=True)

        self.comb_iter_0_left = BranchSeparables(nf, nf, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(nf, nf, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(nf, nf, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(nf, nf, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(nf, nf, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)

        x_relu = self.relu(x_conv0)
        # path 1
        x_path1 = self.path_1(x_relu)
        # path 2
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        # final path
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right, factorized_reduction=False):
        super(NormalCell, self).__init__()
        self.factorized_reduction = factorized_reduction

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        if self.factorized_reduction:
            self.relu = nn.ReLU()
            self.path_1 = nn.Sequential()
            self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
            self.path_1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
            self.path_2 = nn.ModuleList()
            self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
            self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
            self.path_2.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
            self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=0.001, momentum=0.1, affine=True)
        else:
            self.conv_prev_1x1 = nn.Sequential()
            self.conv_prev_1x1.add_module('relu', nn.ReLU())
            self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
            self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))


        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        if self.factorized_reduction:
            x_relu = self.relu(x_prev)
            # path 1
            x_path1 = self.path_1(x_relu)

            # path 2
            x_path2 = self.path_2.pad(x_relu)
            x_path2 = x_path2[:, :, 1:, 1:]
            x_path2 = self.path_2.avgpool(x_path2)
            x_path2 = self.path_2.conv(x_path2)
            # final path
            x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        else:
            x_left = self.conv_prev_1x1(x_prev)

        x_right = self.conv_1x1(x)


        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right, pad=False):
        super(ReductionCell, self).__init__() 
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))

        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, bias=False, reduction=pad)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, bias=False, reduction=pad)
        
        self.comb_iter_1_left = MaxPool(pad=pad)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, bias=False, reduction=pad)

        self.comb_iter_2_left = AvgPool(pad=pad)
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, bias=False, reduction=pad)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False, reduction=pad)
        self.comb_iter_4_right = MaxPool(pad=pad)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NASNet(nn.Module):

    def __init__(self, num_conv_filters, filter_scaling_rate, num_classes,
            num_cells, stem_multiplier, stem):
        super(NASNet, self).__init__()
        self.num_classes = num_classes
        self.num_cells = num_cells
        self.stem = stem

        stem_filters = 32*stem_multiplier
        if self.stem == 'imagenet':
            self.conv0 = nn.Sequential()
            self.conv0.add_module('conv', nn.Conv2d(in_channels=3,
                out_channels=stem_filters, kernel_size=3, padding=0, stride=2,
                bias=False))
            self.conv0.add_module('bn', nn.BatchNorm2d(stem_filters, eps=0.001,
                momentum=0.1, affine=True))

            self.cell_stem_0 = CellStem0(num_conv_filters, stem_multiplier)
            self.cell_stem_1 = CellStem1(num_conv_filters, stem_multiplier)
        elif self.stem == 'cifar':
            self.conv0 = nn.Sequential()
            self.conv0.add_module('conv', nn.Conv2d(in_channels=3,
                out_channels=stem_filters, kernel_size=3, padding=1, stride=2,
                bias=False))
            self.conv0.add_module('bn', nn.BatchNorm2d(stem_filters, eps=0.001,
                momentum=0.1, affine=True))           
        else:
            raise ValueError("Don't know what type of stem %s is."%stem)

        self.block1 = []
        nf, fs = num_conv_filters, filter_scaling_rate
        cell_idx = 0
        self.cell_0 = NormalCell(
                in_channels_left=nf if self.stem == 'imagenet' else 3,
                out_channels_left=nf//fs,
                in_channels_right=nf*fs if self.stem == 'imagenet' else nf*stem_multiplier,
                out_channels_right=nf,
                factorized_reduction=True)
        self.block1.append(self.cell_0)
        in_ch, out_ch = nf*(fs*3), nf
        cells_per_block = num_cells//3
        for i in range(cells_per_block-1):
            cell_idx += 1
            if i==0 and self.stem=='imagenet':
                ch_left = nf*fs if i == 0 else in_ch
            elif i==0 and self.stem=='cifar':
                ch_left = nf*stem_multiplier
            else:
                ch_left = in_ch
            next_cell = NormalCell(in_channels_left=ch_left,
                    out_channels_left=nf,
                    in_channels_right=in_ch,
                    out_channels_right=out_ch)
            # hack to not break sanity check
            setattr(self, "cell_%i"%cell_idx, next_cell)
            self.block1.append(next_cell)

        out_ch = nf*fs
        self.reduction_cell_0 = ReductionCell(in_channels_left=in_ch, out_channels_left=out_ch,
                                              in_channels_right=in_ch, out_channels_right=out_ch,
                                              pad=True)

        cell_idx += 1
        next_cell = NormalCell(in_channels_left=in_ch, out_channels_left=out_ch//fs,
                               in_channels_right=in_ch+nf*fs, out_channels_right=out_ch,
                               factorized_reduction=True)
        setattr(self, "cell_%i"%cell_idx, next_cell)
        in_ch = nf*(fs*6)
        for i in range(cells_per_block-1):
            cell_idx += 1
            next_cell = NormalCell(in_channels_left=nf*fs*4 if i == 0 else in_ch, out_channels_left=out_ch,
                                 in_channels_right=in_ch, out_channels_right=out_ch)
            setattr(self, "cell_%i"%cell_idx, next_cell)
            self.block1.append(next_cell)


        out_ch = nf*fs*2
        self.reduction_cell_1 = ReductionCell(in_channels_left=in_ch, out_channels_left=out_ch,
                                              in_channels_right=in_ch, out_channels_right=out_ch)

        cell_idx += 1
        next_cell = NormalCell(in_channels_left=in_ch, out_channels_left=out_ch//fs,
                               in_channels_right=in_ch+nf*fs*2, out_channels_right=out_ch, 
                               factorized_reduction=True)
        setattr(self, "cell_%i"%cell_idx, next_cell)

        in_ch = nf*(fs*12)
        for i in range(cells_per_block-1):
            cell_idx += 1
            next_cell = NormalCell(in_channels_left=nf*fs*8 if i == 0 else in_ch, out_channels_left=out_ch,
                                      in_channels_right=in_ch, out_channels_right=out_ch)
            setattr(self, "cell_%i"%cell_idx, next_cell)
            self.block1.append(next_cell)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(in_ch, self.num_classes)

    def features(self, input):
        x_conv0 = self.conv0(input)
        if self.stem == 'imagenet':
            x_stem_0 = self.cell_stem_0(x_conv0)
            x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
            cell_stack = [x_stem_1, x_stem_0]
        else:
            cell_stack = [x_conv0, input]

        cell_idx = 0
        for i in range(self.num_cells//3):
            next_cell = getattr(self, "cell_%i"%cell_idx)
            next_out = next_cell(*cell_stack[:2])
            cell_stack = [next_out] + cell_stack
            cell_idx += 1

        x_reduction_cell_0 = self.reduction_cell_0(*cell_stack[:2])
        cell_stack = [x_reduction_cell_0] + cell_stack

        for i in range(self.num_cells//3):
            next_cell = getattr(self, "cell_%i"%cell_idx)
            next_out = next_cell(*cell_stack[:2])
            cell_stack = [next_out] + cell_stack
            cell_idx += 1

        x_reduction_cell_1 = self.reduction_cell_1(*cell_stack[:2])
        cell_stack = [x_reduction_cell_1] + cell_stack

        for i in range(self.num_cells//3):
            next_cell = getattr(self, "cell_%i"%cell_idx)
            next_out = next_cell(*cell_stack[:2])
            cell_stack = [next_out] + cell_stack
            cell_idx += 1

        return cell_stack[0]

    def logits(self, features):
        x = self.relu(features)
        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


class NASNetALarge(NASNet):
    def __init__(self, num_classes=1001):
        super(NASNetALarge, self).__init__(num_conv_filters=168,
                filter_scaling_rate=2, num_classes=num_classes, num_cells=18,
                stem_multiplier=3, stem='imagenet')


class NASNetAMobile(NASNet):
    def __init__(self, num_classes=1001):
        super(NASNetAMobile, self).__init__(num_conv_filters=44,
                filter_scaling_rate=2, num_classes=num_classes, num_cells=12,
                stem_multiplier=1, stem='imagenet')


class NASNetAcifar(NASNet):
    def __init__(self, num_classes=10):
        super(NASNetAcifar, self).__init__(num_conv_filters=32,
                filter_scaling_rate=2, num_classes=num_classes, num_cells=18,
                stem_multiplier=3, stem='cifar')


def nasnetalarge(num_classes=1001, pretrained='imagenet'):
    r"""NASNetALarge model architecture from the
    `"NASNet" <https://arxiv.org/abs/1707.07012>`_ paper.
    """
    if pretrained:
        settings = pretrained_settings['nasnetalarge'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = NASNetALarge(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(model.last_linear.in_features, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']

        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = NASNetALarge(num_classes=num_classes)
    return model


def nasnetamobile(num_classes=1001, pretrained='imagenet'):
    r"""NASNetAMobile model architecture from the
    `"NASNet" <https://arxiv.org/abs/1707.07012>`_ paper.
    """
    assert False
    if pretrained:
        settings = pretrained_settings['nasnetalarge'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = NASNetALarge(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(model.last_linear.in_features, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']

        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = NASNetALarge(num_classes=num_classes)
    return model


if __name__ == "__main__":
    model = NASNetALarge()
    model.eval()

    # checks output hasn't changed from original version

    # saved with original
    #torch.save(model.state_dict(), "nasnet_state_dict.torch")
    state_dict = torch.load("nasnet_state_dict.torch")
    model.load_state_dict(state_dict)

    input = Variable(torch.randn(2,3,331,331))
    output = model(input)
    #torch.save(output, "nasnet_out.torch")
    ref_output = torch.load("nasnet_out.torch")
    print(torch.abs(output-ref_output).sum())
    assert torch.abs(output-ref_output).sum().data.numpy() < 1e-4
    print(output.size())

    # sanity of mobile version of the network
    model = NASNetAMobile()
    model.eval()
    output = model(input)
    print(output.size())

    # sanity of cifar network
    model = NASNetAcifar()
    model.eval()
    input = Variable(torch.randn(2,3,32,32))
    output = model(input)
    print(output.size())

    # check speed of large network
    model = NASNetALarge()
    model.eval().cuda()
    
    import time
    avg = 0.0
    for i in range(10):
        before = time.time()
        input = Variable(torch.randn(1,3,331,331)).cuda()
        output = model(input)
        elapsed = time.time() - before
        avg += elapsed/10.
    print("Average executation time %.2f per minibatch"%avg)
