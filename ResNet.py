import torch.nn as nn


def Conv2d_BN(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
    '''
    conv -> bn -> activate
    '''
    stack = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return stack

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        '''

        :param block: BasicBlock or BottleBlock
        :param layers: numbers of block in each stage([n1,n2,n3,n4])
        '''
        super(ResNet, self).__init__()
        self.stem = nn.Sequential(
            # input: 3*224*224 output: 64*112*112
            # in_channels, out_channels, kernel_size, stride, padding
            Conv2d_BN(3, 64, 7, stride=2, padding=3, bias=False),
            # input: 64*112*112 output: 64*56*56
            # kernel_size, stride, padding
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        in_c = 64
        self.layer1 = self._make_layer(in_c, block, layers[0], first_layer=True)
        self.layer2 = self._make_layer(in_c, block, layers[1])
        in_c *= 2
        self.layer3 = self._make_layer(in_c, block, layers[2])
        in_c *= 2
        self.layer4 = self._make_layer(in_c, block, layers[3])

        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)

        # (batch X feature_nums)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


    def _make_layer(self, in_c, block, layer_num, first_layer=False):
        '''
        generate layer use block

        :param in_c number of input channel
        :param block:
        :param layer_num: number of block in layer
        :return: stack nn
        '''
        layer_nn = []
        # stack block
        # first layer, first stage don't need downsample
        downsample = True
        if first_layer:
            downsample = False
        layer_nn.append(block(in_c, downsample=downsample))

        if downsample:
            in_c *= 2
        for _ in range(1, layer_num):
            layer_nn.append(block(in_c, downsample=False))

        return nn.Sequential(*layer_nn)

class BasicBlock(nn.Module):

    def __init__(self, in_c, downsample=False):
        '''
        init BasicBlock

        :param in_c: number of input channel
        :param downsample: True if need downsample in first stage
        '''
        super(BasicBlock, self).__init__()
        self.downsample = downsample

        if downsample:
            # channel * 2, size / 2
            out_c = in_c * 2
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c)
            )
            self.conv1 = Conv2d_BN(in_c, out_c, 3, stride=2, padding=1)
        else:
            out_c = in_c
            self.conv1 = Conv2d_BN(in_c, out_c, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()


    def forward(self, x):
        '''
        conv_bn 3*3 -> conv_bn 3*3

        :param x: input
        '''
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = x + residual
        x = self.relu(x)

        return x


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

