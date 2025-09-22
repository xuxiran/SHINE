import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import math
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, CTCLoss, AvgPool1d
import random
from torch.nn.parameter import Parameter
import os.path as op
import torch.nn.init as init

class BM(nn.Module):
    def __init__(self, in_dim=306, att_out_dim=64, out_dim=1, kernel_size=25, en_flag=0, dropout=0):
        super().__init__()
        self.spatialattention1 = nn.Linear(in_dim, in_dim)
        self.spatialattention2 = nn.Linear(in_dim, att_out_dim)

        if en_flag == 0:
            encoder_block = FeatureEncoderBlock_v2
        else:
            raise ValueError("Invalid en_flag")

        self.FeatureEncoder = nn.Sequential(
            encoder_block(1, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
            encoder_block(2, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
            encoder_block(3, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
            encoder_block(4, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
            encoder_block(5, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
            encoder_block(6, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
            #encoder_block(7, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
        )

        self.finconv1 = nn.Conv1d(in_channels=att_out_dim, out_channels=4*att_out_dim, kernel_size=1)
        self.finconv2 = nn.Conv1d(in_channels=4*att_out_dim, out_channels=out_dim, kernel_size=1)

        self.d_conv1 = nn.Sequential(
            # 深度卷积（Depthwise Convolution）
            nn.Conv1d(
                in_channels=att_out_dim,  # 输入通道数（finconv1的输出）
                out_channels=att_out_dim,  # 输出通道数（与输入相同）
                kernel_size=kernel_size,
                padding="same",     # 保持空间维度不变
                groups=att_out_dim,        # 分组数=通道数，实现深度卷积
                bias=False
            ),
            #nn.BatchNorm1d(att_out_dim),
            #nn.GELU(),  # 或 nn.ReLU()

            # 逐点卷积（Pointwise Convolution）
            nn.Conv1d(
                in_channels=att_out_dim,
                out_channels=att_out_dim,  # 可调整输出通道数
                kernel_size=1,                # 1x1卷积
                bias=False
            ),
            nn.BatchNorm1d(att_out_dim),
            nn.GELU()
        )
        self.d_conv2 = nn.Conv1d(
                in_channels=1,
                out_channels=1,  # 可调整输出通道数
                kernel_size=1,                # 1x1卷积
                bias=False
            )
        self.sigmoid = nn.Sigmoid()  # ensure output in [0, 1]
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self._initialize_weights()

    def _initialize_weights(self):
        def init_func(module):
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                # 适合带GELU/ReLU激活的卷积层
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                # Transformer 线性层常用Xavier均匀初始化
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                # 归一化层权重初始化为1，偏置为0
                if module.weight is not None:
                    init.ones_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

            # 激活函数层无参数，一般无需初始化

        self.apply(init_func)  # 递归初始化所有子模块

    def forward(self, x):
        # x: [batch, 306, 2500] -> [batch, 2500, att_out_dim]
        x = x.transpose(1, 2)  # [batch, 2500, 306]
        x = self.spatialattention1(x)
        x = self.spatialattention2(x)
        #x = self.spatialattention3(x)
        x = x.transpose(1, 2)  # [batch, att_out_dim, 2500]
        x = self.FeatureEncoder(x)  # [batch, att_out_dim, 2500]
        x = self.d_conv1(x)
        x = x.transpose(1, 2)

        return x


# Define encoder block
class FeatureEncoderBlock(nn.Module):
    def __init__(self, k, channels=[64,64,64,128], kernel_size=25, dropout=0):
        super().__init__()
        self.k = k
        if kernel_size == 3:
            dilation1 = int(pow(2, (2 * self.k) % 5))
            dilation2 = int(pow(2, (2 * self.k + 1) % 5))
        else:
            dilation1 = int(pow(2, (self.k) % 4))
            dilation2 = int(pow(2, (self.k + 1) % 4))
        self.conv1 = nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, padding='same',
                               dilation=dilation1)
        self.conv2 = nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernel_size, padding='same', dilation=dilation2)
        self.conv3 = nn.Conv1d(in_channels=channels[2], out_channels=channels[3], kernel_size=kernel_size, padding='same', dilation=2)
        self.batch_norm1 = nn.BatchNorm1d(num_features=channels[1], eps=1e-05, momentum=0.1)
        self.batch_norm2 = nn.BatchNorm1d(num_features=channels[2], eps=1e-05, momentum=0.1)
        self.GELU = nn.GELU()
        self.GLU = nn.GLU(dim=-2)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x):
        if self.k == 1:
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv3(x)
            x = self.GLU(x)
            x = self.dropout(x)
            return x
        else:
            # residual connection
            input = x
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv3(x)
            x = self.GLU(x)
            x = self.dropout(x)
            return input + x
        
class FeatureEncoderBlock_v2(nn.Module):
    def __init__(self, k=1, channels=[64,64,64,128], kernel_size=25, dropout=0):
        super().__init__()
        self.k = k
        if kernel_size == 3:
            dilation1 = int(pow(2, (2 * self.k) % 5))
            dilation2 = int(pow(2, (2 * self.k + 1) % 5))
        else:
            dilation1 = int(pow(2, (self.k) % 5))
            dilation2 = int(pow(2, (self.k + 1) % 5))
        self.conv1 = nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, padding='same',
                               dilation=dilation1)
        self.conv2 = nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernel_size, padding='same', dilation=dilation2)
        self.conv3 = nn.Conv1d(in_channels=channels[2], out_channels=channels[3], kernel_size=kernel_size, padding='same', dilation=2)
        self.batch_norm1 = nn.BatchNorm1d(num_features=channels[1], eps=1e-05, momentum=0.1)
        self.batch_norm2 = nn.BatchNorm1d(num_features=channels[2], eps=1e-05, momentum=0.1)
        self.GELU = nn.GELU()
        self.GLU = nn.GLU(dim=-2)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x):
        if self.k == 1:
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv3(x)
            x = self.GLU(x)
            x = self.dropout(x)
            return x
        else:
            # residual connection
            input = x
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv3(x)
            x = self.GLU(x)
            x = self.dropout(x)
            return input + x

# Define encoder block
class SpatialEncoderBlock(nn.Module):
    def __init__(self, k, channels, kernel_size=10):
        super().__init__()
        self.k = k

        dilation1 = int(pow(2, (2 * self.k) % 5))
        dilation2 = int(pow(2, (2 * self.k + 1) % 5))

        self.conv1 = nn.Conv1d(in_channels=channels[0], out_channels=channels[1],
                               kernel_size=kernel_size, padding='same', stride=1,
                               dilation=dilation1)
        self.conv2 = nn.Conv1d(in_channels=channels[1], out_channels=channels[2],
                               kernel_size=kernel_size,  padding='same', stride=1,
                               dilation=dilation2)
        self.conv3 = nn.Conv1d(in_channels=channels[2], out_channels=channels[3],
                               kernel_size=kernel_size,  padding='same', stride=1,
                               dilation=2)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm1d(num_features=channels[1], eps=1e-05, momentum=0.1)
        self.batch_norm2 = nn.BatchNorm1d(num_features=channels[2], eps=1e-05, momentum=0.1)
        self.GELU = nn.GELU()
        self.GLU = nn.GLU(dim=-2)
        self.dropout = nn.Dropout1d(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.batch_norm1(x)
        x = self.GELU(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.batch_norm2(x)
        x = self.GELU(x)
        x = self.dropout(x)
        x = self.conv3(x)
        # x = self.max_pool(x)
        x = self.GLU(x)
        x = self.dropout(x)
        return x

class SpatialEncoderBlock_1(nn.Module):
    def __init__(self, kernel_size=10):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1,
                               kernel_size=kernel_size, padding='same', stride=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1,
                               kernel_size=kernel_size,  padding='same', stride=1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1,
                               kernel_size=kernel_size,  padding='same', stride=1)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=1,
                               kernel_size=kernel_size,  padding='same', stride=1)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.GELU = nn.GELU()
        self.dropout = nn.Dropout1d(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.GELU(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.GELU(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.max_pool(x)
        x = self.GELU(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.max_pool(x)
        x = self.GELU(x)
        x = self.dropout(x)

        return x

class SpatialEncoderBlock_2(nn.Module):
    def __init__(self, in_dim, att_out_dim):
        super().__init__()

        self.linear1 = nn.Linear(in_features=in_dim, out_features=int(in_dim / 4))
        self.linear2 = nn.Linear(in_features=int(in_dim / 4), out_features=att_out_dim)
        self.GELU = nn.GELU()
        self.dropout = nn.Dropout1d(p=0.5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.GELU(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.GELU(x)
        x = self.dropout(x)

        return x


class GPTEmbInput(nn.Module):
    def __init__(self, gpt, layer):
        super().__init__()
        self.gpt_layers = gpt.transformer.h[layer::]
        self.lm_head = gpt.lm_head

    def forward(self, inputs_embeds, attention_mask):
        hidden_states = inputs_embeds
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        for i, layer_module in enumerate(self.gpt_layers):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

import torch

if __name__ == '__main__':

    model = BM()
    x = torch.rand(4, 306, 1000)  # [batch_size, in_channels, sequence_length]
    y = model(x)
    print(y.shape)
