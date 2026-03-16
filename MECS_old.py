import torch
import torch.nn as nn
import torch.nn.functional as F


def global_median_pooling(x):
    """全局中位数池化"""
    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled


class ChannelAttention_VersionA(nn.Module):
    """【版本 A：先 Sigmoid，再相加。权重范围 [0, 3]】"""

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention_VersionA, self).__init__()
        self.fc1 = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, inputs):
        avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        median_pool = global_median_pooling(inputs)

        # 1. Avg 路径 (包含 Sigmoid)
        avg_out = self.fc2(F.relu(self.fc1(avg_pool), inplace=True))
        avg_out = torch.sigmoid(avg_out)

        # 2. Max 路径 (包含 Sigmoid)
        max_out = self.fc2(F.relu(self.fc1(max_pool), inplace=True))
        max_out = torch.sigmoid(max_out)

        # 3. Median 路径 (包含 Sigmoid)
        median_out = self.fc2(F.relu(self.fc1(median_pool), inplace=True))
        median_out = torch.sigmoid(median_out)

        # 4. 最后相加 (完全符合架构图)
        out = avg_out + max_out + median_out
        return out


class MECS_VersionA(nn.Module):
    def __init__(self, in_channels, out_channels, channel_attention_reduce=4):
        super(MECS_VersionA, self).__init__()
        assert in_channels == out_channels, "Input and output channels must be the same"

        self.channel_attention = ChannelAttention_VersionA(
            input_channels=in_channels,
            internal_neurons=max(1, in_channels // channel_attention_reduce)
        )

        self.initial_depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.depth_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels),
        ])

        # 修复了权重共享问题的三个独立卷积层
        self.pre_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.spatial_att_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.post_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        # 1. 预处理
        x = self.pre_conv(inputs)
        x = self.act(x)

        # 2. 通道注意力
        channel_att_vec = self.channel_attention(x)
        x_ca = channel_att_vec * x

        # 3. 空间注意力
        initial_out = self.initial_depth_conv(x_ca)
        spatial_outs = [conv(initial_out) for conv in self.depth_convs]
        spatial_out = sum(spatial_outs)

        # 补齐了架构图中的残差连接
        spatial_out = spatial_out + x_ca

        # 补齐了空间 Sigmoid
        spatial_att = torch.sigmoid(self.spatial_att_conv(spatial_out))
        out = spatial_att * x_ca

        # 4. 后处理
        out = self.post_conv(out)
        return out
