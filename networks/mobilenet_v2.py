import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from networks.inverted_residual import InvertedResidual


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class MobileNetV2(nn.Module):
    def __init__(self, input_size=128, width_mult=1., points_num=68):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        # assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.features = nn.Sequential(*self.features)
        # building last several layers
        # make it nn.Sequential
        self.last_block = conv_1x1_bn(input_channel, self.last_channel)

        # building classifier
        self.conv1_after_mbnet = nn.Conv2d(1280,64,(1,1))
        self.conv_node1 = nn.Conv2d(320,128,(1,1))
        self.conv_node2 = nn.Conv2d(1280,128,(1,1))
        self.prelu = nn.PReLU()
        self.fc_landmark = nn.Linear(320, points_num*2)
        self._initialize_weights()

    def forward(self, images):

        x = self.features(images)     # C = 320
        node1 = self.conv_node1(x)   # 1x1 conv 320-->128
        node1 = node1.mean(3).mean(2)  # avg pool

        x = self.last_block(x)        # 1x1 conv 320 -->1280
        node2 = self.conv_node2(x)    # 1x1 conv 1280-->128
        node2 = node2.mean(3).mean(2)  # avg pool

        x = F.avg_pool2d(x, (4, 4))    # avg pool
        x = self.conv1_after_mbnet(x)  # 1x1 conv 1280 --> 64
        x = torch.flatten(x, start_dim=1, end_dim=3)

        final = self.prelu(x)
        end = torch.cat([node1, node2, final], dim=1)
        landmark = self.fc_landmark(end)
        return landmark

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
