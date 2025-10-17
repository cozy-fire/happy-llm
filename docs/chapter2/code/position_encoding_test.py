import math
from dataclasses import dataclass

import torch
from torch import nn

@dataclass
class ModelArgs:
    block_size: int  # 文本序列长度
    n_embd: int # 词向量维度
    dropout: float = 0.1

class PositionalEncoding(nn.Module):
    '''位置编码模块'''

    def __init__(self, args: ModelArgs):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        # self.dropout = nn.Dropout(p=args.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算 theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        print('pe位置编码向量：')
        print(pe)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x

# 设置随机种子以确保结果可重现
torch.manual_seed(42)

# 定义参数
seq_len = 4  # Key/Value序列长度
d_model = 8  # 特征维度

# 模拟 数据源 X
X = torch.randn(seq_len, d_model)
print('文本矩阵 X：')
print(X)

args = ModelArgs(seq_len, d_model)
wpe = PositionalEncoding(args)
pos_x = wpe(X)
print('添加位置编码后的文本矩阵 X：')
print(pos_x)


