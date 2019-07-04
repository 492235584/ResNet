import torch.nn as nn

class BasicBlock(nn.Module):
    expansions = 1  # how many time increase between first convolution and last convolution

    def __init__(self, in_c, out_c, downsample, stride=1):
        '''
        init BasicBlock

        :param in_c: number of input channel
        :param out_c: number of output channel
        :param downsample: convolution of downsample
        :param stride: stride in the first conv
        '''
        super(BasicBlock, self).__init__()
        self.downsample = downsample

        self.conv1 = Conv2d_BN(in_c, out_c, 3, stride=stride, padding=1)
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


class BottleBlock(nn.Module):
    expansions = 4  # how many time increase between first convolution and last convolution

    def __init__(self, in_c, out_c, downsample, stride=1):
        '''
        init BottleBlock

        :param in_c: number of input channel
        :param out_c: number of output channel
        :param downsample: convolution of downsample
        :param stride: stride in the first conv
        '''
        super(BottleBlock, self).__init__()
        self.downsample = downsample

        bollte_c = out_c // 4
        self.conv1 = Conv2d_BN(in_c, bollte_c, 1, stride=stride, bias=False)
        self.conv2 = Conv2d_BN(bollte_c, bollte_c, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(bollte_c, out_c, 1, stride=1, bias=False)
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
        x = self.conv3(x)
        x = self.bn(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = x + residual
        x = self.relu(x)

        return x

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

        self.layer1 = self._make_layer(64, block, layers[0], first_layer=True)
        self.layer2 = self._make_layer(128, block, layers[1])
        self.layer3 = self._make_layer(256, block, layers[2])
        self.layer4 = self._make_layer(512, block, layers[3])

        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansions, num_classes)

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


    def _make_layer(self, fisrt_c, block, layer_num, first_layer=False):
        '''
        generate layer use block

        :param fisrt_c number of first convolution output channel in the paper
        :param block:
        :param layer_num: number of block in layer
        :return: stack nn
        '''
        layer_nn = []
        # stack block
        # if input channel don's equals output channel, need downsample
        downsample = None

        # first block
        if block == BasicBlock:
            if first_layer:
                in_c = fisrt_c
                layer_nn.append(block(in_c, fisrt_c * block.expansions, downsample=downsample, stride=1))
            else:
                in_c = fisrt_c // 2
                downsample = nn.Sequential(
                    nn.Conv2d(in_c, fisrt_c * block.expansions, 1, stride=2, bias=False),
                    nn.BatchNorm2d(fisrt_c * block.expansions)
                )
                layer_nn.append(block(in_c, fisrt_c * block.expansions, downsample=downsample, stride=2))
        elif block == BottleBlock:
            # BottleBlock must down sample in first block
            if first_layer:
                in_c = fisrt_c
                downsample =  nn.Sequential(
                    nn.Conv2d(in_c, fisrt_c * block.expansions, 1, stride=1, bias=False),
                    nn.BatchNorm2d(fisrt_c * block.expansions)
                )
                layer_nn.append(block(in_c, fisrt_c * block.expansions, downsample=downsample, stride=1))
            else:
                in_c = fisrt_c * 2
                downsample = nn.Sequential(
                nn.Conv2d(in_c, fisrt_c * block.expansions, 1, stride=2, bias=False),
                nn.BatchNorm2d(fisrt_c * block.expansions)
                )
                layer_nn.append(block(in_c, fisrt_c * block.expansions, downsample=downsample, stride=2))

        # other block's input channel equals output channel
        for _ in range(1, layer_num):
                layer_nn.append(block(fisrt_c * block.expansions, fisrt_c * block.expansions, downsample=False))

        return nn.Sequential(*layer_nn)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
def ResNet50():
    return ResNet(BottleBlock, [3, 4, 6, 3])

