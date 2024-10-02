import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        output = self.conv1(input)
        output = self.batch_norm(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.batch_norm(output)
        output = self.relu(output)

        return output


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, input):
        return self.net(input)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, y):
        x = self.upsample(x)

        diff_h = x.shape[2] - y.shape[2]
        diff_w = x.shape[3] - y.shape[3]

        pad_left = diff_w // 2
        pad_right = diff_w - pad_left
        pad_top = diff_h // 2
        pad_bottom = diff_h - pad_top

        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        output = torch.cat([x, y], dim=1)
        output = self.conv(output)

        return output


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # self.inc = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        self.inc = DoubleConv(24, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # self.down5 = Down(512, 512)

        self.up4 = Up(1024, 256)

        self.up3 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up1 = Up(128, 64)
        # self.up0 = Up(64, 16)

        self.dec = nn.Conv2d(64, 1, kernel_size=3, padding=1)


    def forward(self, input):
        # print("forward")
        d1 = self.inc(input)
        d2 = self.down1(d1)
        # print(f"d2.shape : {d2.shape}")
        d3 = self.down2(d2)
        # print(f"d3.shape : {d3.shape}")
        d4 = self.down3(d3)
        # print(f"d4.shape : {d4.shape}")
        d5 = self.down4(d4)
        # print(f"d5.shape : {d5.shape}")
        # d6 = self.down5(d5)
        # print(f"d6.shape : {d6.shape}")
        u4 = self.up4(d5, d4)
        # print(f"u4.shape : {u4.shape}")
        u3 = self.up3(u4, d3)
        # print(f"u3.shape : {u3.shape}")
        u2 = self.up2(u3, d2)
        # print(f"u2.shape : {u2.shape}")
        u1 = self.up1(u2, d1)
        # print(f"u1.shape : {u1.shape}")
        # u0 = self.up0(u1, d1)
        # print(f"u0.shape : {u0.shape}")

        return self.dec(u1)

