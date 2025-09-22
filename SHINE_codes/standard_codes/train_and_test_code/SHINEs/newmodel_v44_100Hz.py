import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from BM import BM

in_channel = 96

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class Satt(nn.Module):
    def __init__(self, embed_size=in_channel, heads=8):
        super(Satt, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

    def forward(self, E):
        # In self-attention, values, keys, and queries are typically the same
        M_s = self.attention(E, E, E)
        return M_s


class Extractor(nn.Module):
    def __init__(self, input_channels=in_channel*2):
        super(Extractor, self).__init__()
        self.convs1 = nn.Conv1d(in_channel*3, in_channel, 1)
        self.convt1 = nn.Conv1d(in_channel*4, in_channel*4, 12,groups=in_channel*4)

        self.convs2 = nn.Conv1d(in_channel*4, in_channel, 1)
        self.convt2 = nn.Conv1d(in_channel*4, in_channel*4, 12,groups=in_channel*4)

        self.convs3 = nn.Conv1d(in_channel*4, in_channel, 1)
        self.convt3 = nn.Conv1d(in_channel*4, in_channel*4, 12,groups=in_channel*4)

        self.convs4 = nn.Conv1d(in_channel*4, in_channel, 1)
        self.convt4 = nn.Conv1d(in_channel*4, in_channel*4, 12,groups=in_channel*4)

        self.conv5 = nn.Conv1d(in_channel*4, in_channel*2, 12)

        self.norm1 = nn.LayerNorm(in_channel)
        self.norm2 = nn.LayerNorm(in_channel*2)
        self.norm3 = nn.LayerNorm(in_channel*4)

        self.relu = nn.LeakyReLU()
        self.pad = nn.ZeroPad2d((0, 0, 0, 11))

    def forward(self, x):
        eeg = x

        x = x.permute(0, 2, 1)
        x = self.convs1(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt1(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.convs2(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt2(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.convs3(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt3(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.convs4(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt4(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.conv5(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pad(x)
        return x

class OutputContext(nn.Module):
    def __init__(self, input_channels=in_channel):
        super(OutputContext, self).__init__()

        self.pad = nn.ZeroPad2d((0, 0, 59, 0))
        self.conv = nn.Conv1d(input_channels, in_channel, 60)
        self.norm = nn.LayerNorm(in_channel)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.pad(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.relu(x)
        return x

class SHINE(nn.Module):
    def __init__(self, nb_blocks=6,emb = in_channel, output_dim=1):
        super(SHINE, self).__init__()
        self.spatialattention1 = nn.Linear(306, 306)
        self.spatialattention2 = nn.Linear(306, in_channel)
        self.extractor = Extractor()
        self.satt = Satt()
        self.output_context = OutputContext()
        self.nb_blocks = nb_blocks
        self.linear_layer = nn.Linear(in_channel*2, in_channel)
        self.fc = nn.Linear(in_channel*2, in_channel)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(in_channel, 1)

        self.relu = nn.LeakyReLU()

        self.BM = BM(att_out_dim=in_channel)
        self._initialize_weights()
        # self.dia = DilatedConvNet()

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
        bm_out = self.BM(x)
        eeg = x.permute(0, 2, 1)
        x = self.spatialattention1(eeg)
        x = self.relu(x)
        eeg = self.spatialattention2(x)
        eeg_out = torch.zeros_like(eeg)
        eeg_out_att = torch.zeros_like(eeg)
        for i in range(self.nb_blocks):

            eeg_out = torch.cat((eeg,eeg_out,eeg_out_att), dim=2)

            eeg_out = self.extractor(eeg_out)

            eeg_out = self.linear_layer(eeg_out)
            eeg_out = self.output_context(eeg_out)

            eeg_out_att = self.satt(eeg_out)
            eeg_out_att = eeg_out_att * eeg_out

        eeg_out = torch.cat((eeg_out, bm_out), dim=2)

        eeg_out = self.fc(eeg_out)
        eeg_out = self.sigmoid(eeg_out)
        eeg_out = self.fc2(eeg_out)

        return eeg_out.squeeze(dim=2)


if __name__ == '__main__':
    model = SHINE()
    x = torch.rand(4, 306, 3000)
    y = model(x)
    print(y.shape)
