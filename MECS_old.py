import torch
import torch.nn as nn
import torch.nn.functional as F


def global_median_pooling(x):
    """将每个通道的空间位置展平后取中位数，并恢复成 [B, C, 1, 1]。"""
    median_pooled = torch.median(x.view(x.size(0), x.size(1), -1), dim=2)[0]
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled


class ChannelAttention_VersionA(nn.Module):
    """用 avg/max/median 三条分支生成通道权重，再把三路结果相加。"""

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention_VersionA, self).__init__()
        self.fc1 = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, inputs):
        # 用三种全局统计量描述每个通道：整体强度、最强响应和中位水平。
        avg_pool = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        max_pool = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        median_pool = global_median_pooling(inputs)

        # 平均池化分支更关注整个通道的整体响应强度。
        avg_out = self.fc2(F.relu(self.fc1(avg_pool), inplace=True))
        avg_out = torch.sigmoid(avg_out)

        # 最大池化分支强调最显著的局部激活。
        max_out = self.fc2(F.relu(self.fc1(max_pool), inplace=True))
        max_out = torch.sigmoid(max_out)

        # 中位数池化分支对异常值更稳，可以补充前两种统计信息。
        median_out = self.fc2(F.relu(self.fc1(median_pool), inplace=True))
        median_out = torch.sigmoid(median_out)

        # 三路权重直接相加，得到最终的通道注意力系数。
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

        # groups=in_channels 表示 depthwise conv，也就是每个通道各自卷积。
        self.initial_depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        # 这 6 个分支用不同方向和尺度的条形卷积捕获多尺度空间信息。
        self.depth_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels),
        ])

        # 三个 1x1 卷积分别负责输入预处理、空间权重生成和输出映射。
        self.pre_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.spatial_att_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.post_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        # 先用 1x1 Conv 做通道重映射，再用 GELU 增加非线性表达。
        x = self.pre_conv(inputs)
        x = self.act(x)

        # 根据预处理后的特征生成通道权重，并将权重乘回原特征。
        channel_att_vec = self.channel_attention(x)
        x_ca = channel_att_vec * x

        # 先用 5x5 depthwise conv 做局部空间建模。
        initial_out = self.initial_depth_conv(x_ca)
        # 再并行经过 6 个大核 depthwise 分支，提取不同感受野下的空间上下文。
        spatial_outs = [conv(initial_out) for conv in self.depth_convs]
        # 将多尺度分支的输出相加，融合横向、纵向和不同尺度的信息。
        spatial_out = sum(spatial_outs)

        # 把 x_ca 残差加回来，让空间分支在增强特征时保留原始响应。
        spatial_out = spatial_out + x_ca

        # 用 1x1 Conv + sigmoid 生成空间权重图，再对 x_ca 做逐元素加权。
        spatial_att = torch.sigmoid(self.spatial_att_conv(spatial_out))
        out = spatial_att * x_ca

        # 最后再做一次 1x1 Conv，得到模块输出。
        out = self.post_conv(out)
        return out
