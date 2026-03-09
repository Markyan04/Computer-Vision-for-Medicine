import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Channel Attention
# 对应图中：
# Global Pooling -> C/r -> ReLU -> C -> Sigmoid
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        mid_channels = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pool(x)  # [B, C, 1, 1]
        w = self.fc1(w)  # [B, C/r, 1, 1]
        w = self.relu(w)
        w = self.fc2(w)  # [B, C, 1, 1]
        w = self.sigmoid(w)
        return w


# -------------------------
# Spatial Attention
# 更贴近“空间权重图”的做法：
# 先在通道维聚合，再卷积生成 [B,1,H,W]
# -------------------------
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 严格按照原文档：只用一个1x1卷积
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 原文档：在通道维度上进行全局池化 (只取平均)
        avg_map = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        m = self.conv(avg_map)  # [B, 1, H, W]
        m = self.sigmoid(m)
        return m


# -------------------------
# MDFA
# 与图和文字更一致的版本
# -------------------------
class MDFA(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, reduction=2):
        super(MDFA, self).__init__()

        # 分支1：1x1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 分支2：3x3 空洞卷积 rate=6
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,
                      padding=6 * rate, dilation=6 * rate, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 分支3：3x3 空洞卷积 rate=12
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,
                      padding=12 * rate, dilation=12 * rate, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 分支4：3x3 空洞卷积 rate=18
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,
                      padding=18 * rate, dilation=18 * rate, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 分支5：全局平均池化分支
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        # 拼接后通道数 = 5 * dim_out
        cat_channels = dim_out * 5

        # 通道注意力
        self.channel_att = ChannelAttention(cat_channels, reduction=reduction)

        # 空间注意力
        self.spatial_att = SpatialAttention()

        # 最终 1x1 降维整合
        self.conv_out = nn.Sequential(
            nn.Conv2d(cat_channels, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # 五个分支
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)

        # 全局分支
        global_feat = F.adaptive_avg_pool2d(x, output_size=1)  # [B, C, 1, 1]
        global_feat = self.branch5_conv(global_feat)
        global_feat = self.branch5_bn(global_feat)
        global_feat = self.branch5_relu(global_feat)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=True)

        # 拼接
        feature_cat = torch.cat([feat1, feat2, feat3, feat4, global_feat], dim=1)

        # 两路注意力
        ca = self.channel_att(feature_cat)  # [B, 5*dim_out, 1, 1]
        sa = self.spatial_att(feature_cat)  # [B, 1, H, W]

        # 分别校准
        channel_refined = feature_cat * ca
        spatial_refined = feature_cat * sa

        # 图里是 Add，不是 max
        fused = channel_refined + spatial_refined

        # 最终输出
        out = self.conv_out(fused)
        return out


if __name__ == "__main__":
    x = torch.randn(3, 32, 64, 64)
    model = MDFA(dim_in=32, dim_out=32)
    y = model(x)
    print("input :", x.shape)
    print("output:", y.shape)
