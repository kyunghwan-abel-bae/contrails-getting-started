import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DoubleConv, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, input):
        return self.net(input)


class Down(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Down, self).__init__()
        print("init")
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(ch_in, ch_out)

    def forward(self, input):
        output = self.pool(input)
        output = self.conv(output)

        return output


class Up(nn.Module):
    def __init__(self, ch_in, ch_out, bilinear=False):
        super(Up, self).__init__()
        print("init")
        if bilinear is True:
            self.exp = nn.Upsample(2)
        else:
            self.exp = nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2)

        self.conv = DoubleConv(ch_in, ch_out)

    def forward(self, input):
        print("forward")



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = DoubleConv(24, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

    def forward(self, input):
        print("forward")

        # x = self.linear(x)
        # x = self.relu(x)

        # x = self.inc(24, 64)

        inc = self.inc(input)
        d1 = self.down1(inc)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # print(d1)
        # print(vars(x))
        print(inc.shape)
        print(d1.shape)
        print(d4.shape)

        quit()

        # down 1
        # down 2
        # down 3
        # down 4

        # up 4
        # up 3
        # up 2
        # up 1



        return x
