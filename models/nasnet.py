'''NASNet in PyTorch.

Paper: Learning Transferable Architectures for Scalable Image Recognition
Link: https://arxiv.org/abs/1707.07012

Using most of the definitions from the port of PNASNet, which is the later
paper. The design is simpler, in that it only requires one cell type, instead
of having a cell specifically for reduction and another for normal processing.
Was relatively easy to redefine methods to inlcude this type of processing.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from pnasnet import PNASNet, SepConv, FactorizedReduction, CellBase, DropPath

class NormalCellA(nn.Module):
    def __init__(self, prev_in_shape, in_shape, planes, stride=1, keep_prob=1.0):
        super(NormalCell1, self).__init__()
        self.base = CellBase(prev_in_shape, in_shape, planes, stride=stride)
        in_planes = planes # if base has done its job
        out_planes = planes
        self.stride = stride
        # Branch 1
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        if stride == 2:
            self.reduction = FactorizedReduction(in_planes, out_planes)
        # Branch 2
        self.sep_conv2 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        self.sep_conv3 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        # Branch 5
        self.sep_conv4 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        self.sep_conv5 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        # Drop path
        self.drop_path = DropPath(p=keep_prob)

    def forward(self, x, prev_x):
        x, prev_x = self.base(x, prev_x)
        x, prev_x = F.relu(x), F.relu(prev_x) if prev_x is not None else prev_x
        # Branch 1
        y1 = self.sep_conv1(x)
        if self.drop_path.keep_prob < 1.0 and self.training:
            y1 = self.drop_path(y1)
        if self.stride == 2:
            identity = self.reduction(x)
        else:
            identity = x
        branches = [y1+identity]
        # Branch 2
        y4 = self.sep_conv2(x)
        if prev_x is not None:
            y3 = self.sep_conv3(prev_x)
            if self.drop_path.keep_prob < 1.0 and self.training:
                y3, y4 = self.drop_path(y3), self.drop_path(y4)
            branches.append(y3 + y4)
        else:
            if self.drop_path.keep_prob < 1.0 and self.training:
                y4 = self.drop_path(y4)
            branches.append(y4)
        # Branch 3
        y5 = F.avg_pool2d(x, kernel_size=3, stride=self.stride, padding=1)       
        if prev_x is not None:
            if self.stride == 2:
                identity2 = self.reduction(prev_x)
            else:
                identity2 = prev_x
            branches.append(y5+identity2)
        else:
            branches.append(y5)
        if prev_x is not None:
            assert False
            # appears in the paper this is two average pooling blocks in parallel, which is nonsense
            # (we suppose this is just what the algorithm spat out, in our lab)
            branches.append(F.avg_pool2d(prev_x, kernel_size=3, stride=self.stride, padding=1))
            y6 = self.sep_conv4(prev_x)
            y7 = self.sep_conv5(prev_x)
            if self.drop_path.keep_prob < 1.0 and self.training:
                y6, y7 = self.drop_path(y6), self.drop_path(y7)
            branches.append(y6+y7)
        return torch.cat(branches, 1)


class NASNet(PNASNet):
    def _downsample(self, planes):
        return NotImplementedError
        prev_in_shape = self.qnd_shape_inference[0].size()
        in_shape = self.qnd_shape_inference[1].size()
        layer = self.cell_type(prev_in_shape, in_shape, planes, stride=2, keep_prob=self.keep_prob)
        _qnd = layer(self.qnd_shape_inference[1], self.qnd_shape_inference[0])
        self.qnd_shape_inference.append(_qnd)
        _ = self.qnd_shape_inference.pop(0)
        return layer

