'''PNASNet in PyTorch.

Paper: Progressive Neural Architecture Search

Note: could not figure out from the slim code what is supposed to happen in
e.g. Cell 3 when an identity transform is added to a convolution, if the
convolution is stride 2. They are obviously going to be the wrong shape and
impossible to cast onto each other. In those cases, I've decided to use the
Factorized Reduction in their code.

Looks like in the original code, paths are dropped potentially whenever a 
convolution is performed. I've limited it to just wherever there is a branch
in a Cell and made them all explicit, just to make things easier to keep 
track of.

In Cell 5 there is a chain of two convolutional blocks. If I naively used
stride=2 with both in reduce blocks the size of the output could never cast
onto the output of the parallel branch. So, I thought it was probably best to
just not apply stride 2 in the second convolution. I don't know what is done in
this case in the original model (it is difficult to figure it out from reading
the code).

There are probably other mistakes of this type, which may or may not matter.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class SepConv(nn.Module):
    '''Separable Convolution: grouped followed by pointwise
    (https://arxiv.org/abs/1610.02357).'''
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(SepConv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes,
                               kernel_size, stride,
                               padding=(kernel_size-1)//2,
                               bias=False, groups=in_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, 1, 1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn2(self.conv2(self.bn1(self.conv1(x))))


class FactorizedReduction(nn.Module):
    """
    Hack to ensure input from previous block is always the same shape as the
    input from the current block.
    """
    def __init__(self, prev_in_planes, planes):
        super(FactorizedReduction, self).__init__()
        # two branches, with avgpool offset in one branch, but not in the other 
        c = prev_in_planes
        self.avgpoolA = nn.AvgPool2d(2)
        self.pointwiseA = nn.Conv2d(c, planes//2, 1)
        self.padB = nn.ZeroPad2d((0,1,0,1))
        self.avgpoolB = nn.AvgPool2d(2)
        self.pointwiseB = nn.Conv2d(c, planes//2, 1)
        # after concat, batchnorm
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(x)
        branchA = self.avgpoolA(out)
        branchA = self.pointwiseA(branchA)
        branchB = self.padB(out)
        branchB = self.avgpoolB(branchB)
        branchB = self.pointwiseB(branchB)
        out = torch.cat([branchA, branchB], 1)
        return self.bn(out)


class DropPath(nn.Module):
    """
    Zeros input x with probability 1-p independently over examples.
    p is the probability of keeping the input, the opposite of the normal
    operation of the Dropout module.
    """

    def __init__(self, p=0.5, inplace=False):
        super(DropPath, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.keep_prob, self.p = p, 1.-p
        self.inplace = inplace

    def forward(self, input):
        if not self.training:
            return input
        batch_size = input.size(0)
        mask = torch.ones(batch_size, 1, 1, 1)
        if input.is_cuda:
            mask = mask.cuda()
        mask = F.dropout(mask, self.p, self.training, self.inplace)
        return mask*input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) \
            + inplace_str + ')'


class CellBase(nn.Module):
    """Implements base transformations to ensure uniform number of
    channels and shape before applying transformations. Not really discussed in
    the paper, but present in the code."""
    def __init__(self, prev_in_shape, in_shape, planes, stride=1):
        super(CellBase, self).__init__()
        # assumed that previous layer has just concatenated everything together
        # for this layer to deal with
        self._make_prev_transform(prev_in_shape, in_shape, planes)
        # then need standard transformation for current
        _, c, h, w = in_shape
        layers = []
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(c, planes, 1))
        layers.append(nn.BatchNorm2d(planes))
        self.current_transform = nn.Sequential(*layers)

    def do_nothing(self, x):
        return x

    def _make_prev_transform(self, prev_in_shape, in_shape, planes):
        if prev_in_shape is None:
            self.prev_transform = self.do_nothing
            return None
        _, pc, ph, pw = prev_in_shape
        _, c, h, w = in_shape
        assert ph == pw and h == w
        if ph != h:
            self.prev_transform = FactorizedReduction(pc, planes)
            return None
        elif pc != planes:
            # prepare sequential pointwise
            layers = []
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(pc, planes, 1))
            layers.append(nn.BatchNorm2d(planes))
            self.prev_transform = nn.Sequential(*layers)
            return None
        else:
            # do nothing
            self.prev_transform = self.do_nothing
            return None
        
    def forward(self, current, prev):
        prev_ch = prev.size(1) if prev is not None else -1
        out_prev = self.prev_transform(prev)
        out_ch = current.size(1)
        out_current = self.current_transform(current)
        if out_prev is not None:
            if out_prev.size(1) == prev_ch and not out_current.size(1) == out_ch:
                import pdb
                pdb.set_trace()
        return out_current, out_prev
    

class Cell1(nn.Module):
    def __init__(self, prev_in_shape, in_shape, planes, stride=1, keep_prob=1.0):
        super(Cell1, self).__init__()
        self.base = CellBase(prev_in_shape, in_shape, planes, stride=stride)
        self.stride = stride
        in_planes = planes # if base has done its job
        out_planes = planes
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        if stride==2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
        self.drop_path = DropPath(p=keep_prob)

    def forward(self, x, prev_x):
        x, prev_x = self.base(x, prev_x)
        x, prev_x = F.relu(x), F.relu(prev_x) if prev_x is not None else prev_x
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            y2 = self.bn1(self.conv1(y2))
        if self.drop_path.keep_prob < 1.0 and self.training:
            y1, y2 = self.drop_path(y1), self.drop_path(y2)
        return y1+y2


class Cell2(nn.Module):
    def __init__(self, prev_in_shape, in_shape, planes, stride=1, keep_prob=1.0):
        super(Cell2, self).__init__()
        self.base = CellBase(prev_in_shape, in_shape, planes, stride=stride)
        in_planes = planes # if base has done its job
        out_planes = planes
        self.stride = stride
        # Left branch
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        self.sep_conv2 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        # Right branch
        self.sep_conv3 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        if stride==2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
        # Reduce channels
        self.conv2 = nn.Conv2d(2*out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        # Drop path
        self.drop_path = DropPath(p=keep_prob)

    def forward(self, x, prev_x):
        x, prev_x = self.base(x, prev_x)
        x, prev_x = F.relu(x), F.relu(prev_x) if prev_x is not None else prev_x
        # Left branch
        y1 = self.sep_conv1(x)
        y2 = self.sep_conv2(x)
        # Right branch
        y3 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            y3 = self.bn1(self.conv1(y3))
        y4 = self.sep_conv3(x)
        # Drop paths
        if self.drop_path.keep_prob < 1.0 and self.training:
            y1, y2, y3, y4 = [self.drop_path(y) for y in [y1,y2,y3,y4]]
        # Concat & reduce channels
        b1 = y1+y2
        b2 = y3+y4
        y = torch.cat([b1,b2], 1)
        return self.bn2(self.conv2(y))


class Cell3(nn.Module):
    def __init__(self, prev_in_shape, in_shape, planes, stride=1, keep_prob=1.0):
        super(Cell3, self).__init__()
        self.base = CellBase(prev_in_shape, in_shape, planes, stride=stride)
        in_planes = planes # if base has done its job
        out_planes = planes
        self.stride = stride
        # Left branch
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        # Middle branch
        self.sep_conv2 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        if self.stride == 2:
            self.reduction = FactorizedReduction(in_planes, out_planes)
        # Right branch
        self.conv_1b7 = nn.Conv2d(in_planes, out_planes, kernel_size=(1,7), stride=(1,stride), padding=(0,3))
        self.conv_7b1 = nn.Conv2d(in_planes, out_planes, kernel_size=(7,1), stride=(stride,1), padding=(3,0))
        # Drop path
        self.drop_path = DropPath(p=keep_prob)

    def forward(self, x, prev_x):
        x, prev_x = self.base(x, prev_x)
        x, prev_x = F.relu(x), F.relu(prev_x) if prev_x is not None else prev_x
        # Left branch
        y1 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        y2 = self.sep_conv1(x)
        branchA = y1+y2
        # Middle branch
        if self.stride == 2:
            y3 = self.sep_conv2(x) 
            y4 = self.reduction(x)
        else:
            y3 = self.sep_conv2(x)
            y4 = x
        if self.drop_path.keep_prob < 1.0 and self.training:
            y3, y4 = self.drop_path(y3), self.drop_path(y4)
        branchB = y3 + y4
        # Right branch
        if prev_x is not None:
            y5 = self.conv_1b7(prev_x)
            y5 = self.conv_7b1(y5)
            y6 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
            # Drop paths
            if self.drop_path.keep_prob < 1.0 and self.training:
                y5, y6 = self.drop_path(y5), self.drop_path(y6)
            branchC = [y5+y6]
        else:
            branchC = [] 
        return torch.cat([branchA, branchB]+branchC, 1)


class Cell4(nn.Module):
    def __init__(self, prev_in_shape, in_shape, planes, stride=1, keep_prob=1.0):
        super(Cell4, self).__init__()
        self.base = CellBase(prev_in_shape, in_shape, planes, stride=stride)
        in_planes = planes # if base has done its job
        out_planes = planes
        self.stride = stride
        # Branch 1
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        # Branch 2
        self.sep_conv2 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        self.sep_conv3 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        # Branch 3
        self.sep_conv4 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)       
        if stride == 2:
            self.reduction = FactorizedReduction(in_planes, out_planes)
        # Branch 4
        self.sep_conv5 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        # Drop path
        self.drop_path = DropPath(p=keep_prob)

    def forward(self, x, prev_x):
        x, prev_x = self.base(x, prev_x)
        x, prev_x = F.relu(x), F.relu(prev_x) if prev_x is not None else prev_x
        # Branch 1
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.drop_path.keep_prob < 1.0 and self.training:
            y1, y2 = self.drop_path(y1), self.drop_path(y2)
        branches = [y1 + y2]
        # Branch 2
        y3 = self.sep_conv2(x)
        y4 = self.sep_conv3(x)
        if self.drop_path.keep_prob < 1.0 and self.training:
            y3, y4 = self.drop_path(y3), self.drop_path(y4)
        branches.append(y3 + y4)
        # Branch 3
        if prev_x is not None:
            y5 = self.sep_conv4(prev_x)
            if self.drop_path.keep_prob < 1.0 and self.training:
                y5 = self.drop_path(y5)
            if self.stride == 2: # identity won't map without doing this
                identity = self.reduction(x) # despite this not really being identity
            else:
                identity = x
            branches.append(y5 + identity)
            # Branch 4
            y6 = self.sep_conv5(prev_x)
            if self.drop_path.keep_prob < 1.0 and self.training:
                y6 = self.drop_path(y6)
            y7 = F.max_pool2d(prev_x, kernel_size=3, stride=self.stride, padding=1)
            branches.append(y6+y7)
        else:
            branches.append(x)
        return torch.cat(branches, 1)


class Cell5(nn.Module):
    def __init__(self, prev_in_shape, in_shape, planes, stride=1, keep_prob=1.0):
        super(Cell5, self).__init__()
        self.base = CellBase(prev_in_shape, in_shape, planes, stride=stride)
        in_planes = planes # if base has done its job
        out_planes = planes
        self.stride = stride
        # This structure is a little nested, but branches are still numbered left to right
        # Branch 1
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        # Branch 2
        self.sep_conv2 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        self.sep_conv3 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        # Branch 3
        self.sep_conv4 = SepConv(in_planes, out_planes, kernel_size=3, stride=1) # see notes at top
        # Branch 4
        self.sep_conv5 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        if stride == 2:
            self.reduction = FactorizedReduction(in_planes, out_planes)
        # Branch 5
        self.sep_conv6 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        # Drop path
        self.drop_path = DropPath(p=keep_prob)

    def forward(self, x, prev_x):
        x, prev_x = self.base(x, prev_x)
        x, prev_x = F.relu(x), F.relu(prev_x) if prev_x is not None else prev_x
        # Branch 1
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.drop_path.keep_prob < 1.0 and self.training:
            y1, y2 = self.drop_path(y1), self.drop_path(y2)
        branches = [y1 + y2]
        # Branch 2
        y3 = self.sep_conv2(x)
        y4 = self.sep_conv3(x)
        if self.drop_path.keep_prob < 1.0 and self.training:
            y3, y4 = self.drop_path(y3), self.drop_path(y4)
        b2 = y3 + y4
        # Branch 3
        y5 = self.sep_conv4(b2)
        y6 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.drop_path.keep_prob < 1.0 and self.training:
            y5, y6 = self.drop_path(y5), self.drop_path(y6)
        branches.append(y5+y6)
        # Branch 4 
        if prev_x is not None:
            y7 = self.sep_conv5(prev_x)
            if self.drop_path.keep_prob < 1.0 and self.training:
                y7 = self.drop_path(y7)
            if self.stride == 2: # identity won't map without doing this
                identity = self.reduction(x) # despite this not really being identity
            else:
                identity = x
            branches.append(y7 + identity)
            # Branch 4
            y8 = self.sep_conv6(prev_x)
            if self.drop_path.keep_prob < 1.0 and self.training:
                y8 = self.drop_path(y8)
            y9 = F.max_pool2d(prev_x, kernel_size=3, stride=self.stride, padding=1)
            branches.append(y8+y9)
        else:
            branches.append(x)
        return torch.cat(branches, 1)


class CIFARStem(nn.Module):
    def __init__(self, num_planes):
        super(CIFARStem, self).__init__()
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_planes)

    def forward(self, x):
        return F.relu(self.bn1(self.conv1(x)))


class AuxHead(nn.Module):
    def __init__(self, qnd_shape_inference, num_classes=10):
        super(AuxHead, self).__init__()
        # aux output to improve convergence (classification shortcut)
        self.pool = nn.AvgPool2d(5, stride=3)
        # local shape inference
        qnd_shape_inference = self.pool(qnd_shape_inference[1])
        in_planes = qnd_shape_inference.size(1)
        self.pointwise = nn.Conv2d(in_planes, 128, 1)
        qnd_shape_inference = self.pointwise(qnd_shape_inference)
        self.pointwise_bn = nn.BatchNorm2d(128)
        _, c, h, w = qnd_shape_inference.size()
        assert h == w # idk what to do if they aren't
        # PNASNet's way of implementing a fc layer is wild
        self.conv2d_fc = nn.Conv2d(128, 728, h)
        #self.conv2d_fc_bn = nn.BatchNorm2d(728)
        #qnd_shape_inference = layers[-1](self.qnd_shape_inference)
        #_, c, h, w = qnd_shape_inference.size()
        self.linear = nn.Linear(728, num_classes)

    def forward(self, x):
        out = self.pool(x)
        out = self.pointwise(out)
        out = self.pointwise_bn(out)
        out = F.relu(out)
        out = self.conv2d_fc(out)
        #out = self.conv2d_fc_bn(out)
        out = F.relu(out)
        return self.linear(out.view(out.size(0),-1))


class CellSequential(nn.Sequential):
    """Sequential with inputs from previous blocks."""
    def forward(self, input, prev_input):
        for module in self._modules.values():
            out_new = module(input, prev_input)
            prev_input = input
            input = out_new
        return [input, prev_input]


class PNASNet(nn.Module):
    def __init__(self, cell_type, num_cells, num_planes, stem_multiplier=3,
            drop_path_keep_prob=0.6):
        super(PNASNet, self).__init__()
        # quick and dirty shape inference
        self.qnd_shape_inference = Variable(torch.randn(1,3,32,32))
        self.cell_type = cell_type
        self.keep_prob = drop_path_keep_prob

        # stem
        self.stem = CIFARStem(num_planes*stem_multiplier)
        self.qnd_shape_inference = [None, self.stem(self.qnd_shape_inference)]

        self.cellseq1 = self._make_layer(num_planes, num_cells=6)
        self.reduce1 = self._downsample(num_planes*2)
        self.cellseq2 = self._make_layer(num_planes*2, num_cells=6)
        self.aux_head  = AuxHead(self.qnd_shape_inference)
        self.reduce2 = self._downsample(num_planes*4)
        self.cellseq3 = self._make_layer(num_planes*4, num_cells=6)

        if self.qnd_shape_inference[1].size(1) != num_planes*4:
            prev_in_shape = self.qnd_shape_inference[0].size()
            in_shape = self.qnd_shape_inference[1].size()
            self.linear_base = CellBase(prev_in_shape, in_shape, num_planes*4)
        self.linear = nn.Linear(num_planes*4, 10)

        # re-initialise weights (batchnorm parameters could get broken by qnd shape inference
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, num_cells):
        layers = []
        prev_in_shape = None
        for _ in range(num_cells):
            in_shape = self.qnd_shape_inference[1].size()
            prev_in_shape = self.qnd_shape_inference[0].size() if self.qnd_shape_inference[0] is not None else None
            layers.append(self.cell_type(prev_in_shape, in_shape, planes, stride=1, keep_prob=self.keep_prob))
            _qnd = layers[-1](self.qnd_shape_inference[1], self.qnd_shape_inference[0])
            self.qnd_shape_inference.append(_qnd)
            _ = self.qnd_shape_inference.pop(0)
        return CellSequential(*layers)

    def _downsample(self, planes):
        prev_in_shape = self.qnd_shape_inference[0].size()
        in_shape = self.qnd_shape_inference[1].size()
        layer = self.cell_type(prev_in_shape, in_shape, planes, stride=2, keep_prob=self.keep_prob)
        _qnd = layer(self.qnd_shape_inference[1], self.qnd_shape_inference[0])
        self.qnd_shape_inference.append(_qnd)
        _ = self.qnd_shape_inference.pop(0)
        return layer

    def forward(self, x):
        out = [None, self.stem(x)]
        out = self.cellseq1(out[-1], out[-2])
        out += [self.reduce1(out[-1], out[-2])]
        _ = out.pop(0)
        out = self.cellseq2(out[-1], out[-2])
        if self.train:
            aux_out = self.aux_head(out[-1])
        out += [self.reduce2(out[-1], out[-2])]
        _ = out.pop(0)
        out = self.cellseq3(out[-1], out[-2])
        if hasattr(self, 'linear_base'):
            out, _ = self.linear_base(out[-1], out[-2])
        else:
            out = out[-1]
        out = F.avg_pool2d(F.relu(out), 8)
        out = self.linear(out.view(out.size(0), -1))
        if self.training:
            return out, aux_out
        else:
            return out # at test time don't care about aux head


def PNASNet1():
    return PNASNet(Cell1, num_cells=6, num_planes=32)

def PNASNet2():
    return PNASNet(Cell2, num_cells=6, num_planes=32)

def PNASNet3():
    return PNASNet(Cell3, num_cells=6, num_planes=32)

def PNASNet4():
    return PNASNet(Cell4, num_cells=6, num_planes=32)

def PNASNet5():
    return PNASNet(Cell5, num_cells=6, num_planes=32)


def test():
    net = PNASNet1()
    net.train()
    #print(net)
    x = Variable(torch.randn(1,3,32,32))
    y = net(x)
    print(y)
    net = PNASNet2()
    y = net(x)
    print(y)
    net = PNASNet3()
    y = net(x)
    print(y)
    net = PNASNet4()
    y = net(x)
    print(y)
    net = PNASNet5()
    y = net(x)
    print(y)


if __name__ == '__main__':
    test()
