# coding=utf-8
import torch.nn as nn


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, kernel_size=3, stride=2):
        super(InvertedResidualBlock, self).__init__()

        if stride != 1 and stride != 2:
            raise ValueError("Stride should be 1 or 2")

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False),  # 扩展通道
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                      kernel_size, stride, 1,
                      groups=in_channels * expansion_factor, bias=False),   # depth-wise卷积操作
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, out_channels, 1,
                      bias=False),                                          # 恢复输出通道
            nn.BatchNorm2d(out_channels))

        self.is_residual = True if stride == 1 else False                   # 当该单元的stide = 1 时采用skip connection
        self.is_conv_res = False if in_channels == out_channels else True   # 匹配输入 输出通道的一致性

        # Assumption based on previous ResNet papers: If the number of filters doesn't match,
        # there should be a conv1x1 operation.
        if stride == 1 and self.is_conv_res:
            self.conv_res = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):           # 前向传播
        block = self.block(x)
        if self.is_residual:
            if self.is_conv_res:
                return self.conv_res(x) + block
            return x + block
        return block


# 该函数分别进行3x3卷积 BN ReLU6操作
def conv2d_bn_relu6(in_channels, out_channels, kernel_size=3, stride=2, dropout_prob=0.0):
    # To preserve the equation of padding. (k=1 maps to pad 0, k=3 maps to pad 1, k=5 maps to pad 2, etc.)
    padding = (kernel_size + 1) // 2 - 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        # For efficiency, Dropout is placed before Relu.
        nn.Dropout2d(dropout_prob, inplace=True),
        # Assumption: Relu6 is used everywhere.
        nn.ReLU6(inplace=True)
    )


def inverted_residual_sequence(in_channels, out_channels, num_units, expansion_factor=6, kernel_size=3, initial_stride=2):
    # 第一个单元stride=initial_stride 后续 stride=1
    bottleneck_arr = [
        InvertedResidualBlock(in_channels, out_channels, expansion_factor, kernel_size, initial_stride)
    ]

    for i in range(num_units-1):
        bottleneck_arr.append(
            InvertedResidualBlock(out_channels, out_channels, expansion_factor, kernel_size, 1))

    return bottleneck_arr


# 建立网络图模型
class MobileNetV2(nn.Module):
    def __init__(self, args):
        super(MobileNetV2, self).__init__()

        # 配置某些block的stride，满足downsampling的需求
        s1, s2 = 2, 2
        if args.downsampling == 16:
            s1, s2 = 2, 1
        elif args.downsampling == 8:
            s1, s2 = 1, 1

        '''
        network_settings网络的相关配置，从该参数可以看出，Mobile-Net由9个部分组成,
        姑且叫做Mobile block。
        network_settings中:
        't'表示Inverted Residuals的扩征系数
        'c'表示该block输出的通道数
        ‘n’表示当前block由几个残差单元组成
        's'表示当前block的stride
        '''
        # Network is created here, then will be unpacked into nn.sequential
        self.network_settings = [{'t': -1, 'c': 32, 'n': 1, 's': s1},
                                 {'t': 1, 'c': 16, 'n': 1, 's': 1},
                                 {'t': 6, 'c': 24, 'n': 2, 's': s2},
                                 {'t': 6, 'c': 32, 'n': 3, 's': 2},
                                 {'t': 6, 'c': 64, 'n': 4, 's': 2},
                                 {'t': 6, 'c': 96, 'n': 3, 's': 1},
                                 {'t': 6, 'c': 160, 'n': 3, 's': 2},
                                 {'t': 6, 'c': 320, 'n': 1, 's': 1},
                                 {'t': None, 'c': 1280, 'n': 1, 's': 1}]
        self.num_classes = args.num_classes

        ###############################################################################################################
        # Feature Extraction part
        # Layer 0
        # args.width_multiplier网络的通道"瘦身"系数
        # block 0
        self.network = [conv2d_bn_relu6(args.num_channels,
                        int(self.network_settings[0]['c'] * args.width_multiplier), args.kernel_size,
                        self.network_settings[0]['s'], args.dropout_prob)]

        # Layers from 1 to 7
        for i in range(1, 8):
            # inverted_residual_sequence 根据当前network_settings[i]的配置建立图模型
            self.network.extend(
                inverted_residual_sequence(
                    int(self.network_settings[i - 1]['c'] * args.width_multiplier),
                    int(self.network_settings[i]['c'] * args.width_multiplier),
                    self.network_settings[i]['n'], self.network_settings[i]['t'],
                    args.kernel_size, self.network_settings[i]['s']))

        # Last layer before flattening
        self.network.append(
            conv2d_bn_relu6(int(self.network_settings[7]['c'] * args.width_multiplier),
                            int(self.network_settings[8]['c'] * args.width_multiplier),
                            1, self.network_settings[8]['s'], args.dropout_prob))

        ###############################################################################################################
        # Classification part
        # 以上输出的特征图进行池化 分类
        self.network.append(nn.Dropout2d(args.dropout_prob, inplace=True))
        self.network.append(nn.AvgPool2d(
            (args.img_height // args.downsampling, args.img_width // args.downsampling)))
        self.network.append(nn.Dropout2d(args.dropout_prob, inplace=True))
        self.network.append(
            nn.Conv2d(int(self.network_settings[8]['c'] * args.width_multiplier), self.num_classes, 1, bias=True))

        self.network = nn.Sequential(*self.network)

        self.initialize()

    def forward(self, x):                   # MobileNetV2的前向传播
        # Debugging mode
        # for op in self.network:
        #     x = op(x)
        #     print(x.shape)
        x = self.network(x)
        x = x.view(-1, self.num_classes)
        return x

    # 初始化权重函数
    def initialize(self):
        """Initializes the model parameters"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
