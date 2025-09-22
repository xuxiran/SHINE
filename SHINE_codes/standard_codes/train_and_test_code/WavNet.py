import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels
        self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation)

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):

        x = input_data
        h = x
        B, C, L = x.shape

        assert C == self.res_channels
        h = self.dilated_conv_layer(h)

        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

        # residual and skip outputs
        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, dilation_cycle):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers

        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels,
                                                       dilation=2 ** (n % dilation_cycle)))

    def forward(self, input_data):
        x = input_data


        h = x
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n](h)  # use the output from last residual layer
            skip = skip + skip_n  # accumulate all skip outputs

        return skip * math.sqrt(1.0 / self.num_res_layers)  # normalize for training stability


class WavNet(nn.Module):
    def __init__(self, input_channels=306, out_channels=1, sub_num=12,
                 res_channels=128, skip_channels=128,
                 num_res_layers=36, dilation_cycle=12,
                 sub_dim=128):
        super(WavNet, self).__init__()

        # initial conv1x1 with relu
        self.spatialattention1 = nn.Linear(306, 306)
        self.spatialattention2 = nn.Linear(306, 128)
        self.relu = nn.LeakyReLU()

        # all residual layers
        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             dilation_cycle=dilation_cycle)

        # final conv1x1 -> relu -> zeroconv1x1
        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        nn.Conv1d(skip_channels, out_channels,kernel_size=1))
        self.proj1 = nn.Linear(res_channels,res_channels)
        self.fc = nn.Linear(128, 64)

    def forward(self, input_data):
        '''
        input_data: (B, L, C)
        '''
        x = input_data
        x = x.transpose(-1,-2)
        x = self.spatialattention1(x)
        x = self.relu(x)
        x = self.spatialattention2(x)

        x = x.transpose(-1,-2)
        x = self.residual_layer(x)

        x = x.transpose(-1,-2)
        x = self.fc(x)


        return x



if __name__ == '__main__':

    net = WavNet(input_channels=306, out_channels=1)
    indata = torch.randn((4, 306, 3000), dtype=torch.float32)

    out = net(indata)

    print(out.shape)
