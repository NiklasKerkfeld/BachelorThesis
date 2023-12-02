"""
Implementation of an 3D UNet with an ResNet based Encoder
"""

import monai.losses
import torch
from monai.networks.nets import BasicUNet
from torch.nn import Module, Conv3d, BatchNorm3d, ConvTranspose3d


class ConvLayer(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, stride: int = 1,
                 activation: bool = True):
        super().__init__()
        self.conv = Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=False)

        self.norm = BatchNorm3d(out_channels)
        self.activation = torch.nn.LeakyReLU() if activation is not None else None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)

        return x


class UpsampleLayer(Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = ConvLayer(in_channels, out_channels=in_channels // 4, kernel_size=3, padding=1)
        self.transpose = ConvTranspose3d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2)

    def forward(self, x, shortcut):
        x = torch.concat([x, shortcut], dim=1)
        x = self.conv(x)
        x = self.transpose(x)
        return x


class BottleneckBlock(Module):
    def __init__(self, in_channels: int, stride=1):
        super().__init__()
        bottleneck = max(in_channels // 4, 1)

        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=bottleneck, kernel_size=1,
                               padding=0)

        self.conv2 = ConvLayer(in_channels=bottleneck, out_channels=bottleneck * stride, kernel_size=3, stride=stride,
                               padding=1)

        self.conv3 = ConvLayer(in_channels=bottleneck * stride, out_channels=in_channels * stride, kernel_size=1,
                               padding=0)

        self.downsample = None if stride == 1 else Conv3d(in_channels, in_channels * stride, kernel_size=3,
                                                          stride=stride, padding=1)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        return x


class Block(Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.layer1 = BottleneckBlock(in_channels, stride=2)
        self.layer2 = BottleneckBlock(in_channels * 2)
        self.layer3 = BottleneckBlock(in_channels * 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class Model(Module):
    def __init__(self, in_channels: int, out_channels:int):
        super().__init__()

        self.conv1 = ConvLayer(in_channels, 16, kernel_size=3, padding=1)

        self.block1 = Block(in_channels=16)
        self.block2 = Block(in_channels=32)
        self.block3 = Block(in_channels=64)
        self.block4 = Block(in_channels=128)

        self.transpose = ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.conv2 = ConvLayer(256, 128, kernel_size=3, padding=1)

        self.up1 = UpsampleLayer(256)
        self.up2 = UpsampleLayer(128)
        self.up3 = UpsampleLayer(64)

        self.conv3 = ConvLayer(32, 32, 3, 1, 1, activation=True)
        self.conv4 = ConvLayer(32, out_channels=out_channels, kernel_size=1, padding=0, stride=1, activation=False)

    def forward(self, x):
        x_0 = self.conv1(x)
        x_1 = self.block1(x_0)
        x_2 = self.block2(x_1)
        x_3 = self.block3(x_2)
        x_4 = self.block4(x_3)

        x_4 = self.transpose(x_4)
        x_4 = self.conv2(x_4)

        x = self.up1(x_4, x_3)
        x = self.up2(x, x_2)
        x = self.up3(x, x_1)

        x = torch.concat([x, x_0], dim=1)

        x = self.conv3(x)
        x = self.conv4(x)

        return x


if __name__ == '__main__':
    model = Model(3, 2)
    basic = BasicUNet(spatial_dims=3, in_channels=3, features=(32, 32, 64, 128, 256, 32))

    print(model)
    loss_fn = monai.losses.DiceLoss(include_background=False, softmax=True, to_onehot_y=True)
    target = torch.randint(0, 2, (8, 1, 64, 64, 32))

    data = torch.randn(1, 3, 224, 224, 32)
    print(f"input.shape={data.shape}")

    pred = model(data)
    print(f"{pred.shape=}")

    pred = basic(data)
    print(f"{pred.shape=}")

    # loss = loss_fn(pred, target)
    # loss.backward(retain_graph=True)
