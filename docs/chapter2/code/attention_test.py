'''注意力计算函数'''
import math
import torch
from torch.onnx.symbolic_opset9 import tensor
import torch.nn.functional as F

def attention(query, key, value, dropout=None):
    '''
    args:
    query: 查询值矩阵
    key: 键值矩阵
    value: 真值矩阵
    '''
    # 获取键向量的维度，键向量的维度和值向量的维度相同
    d_k = query.size(-1)
    # 计算Q与K的内积并除以根号dk
    # transpose——相当于转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Softmax
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        # 采样
     # 根据计算结果对value进行加权求和
    return torch.matmul(p_attn, value), p_attn

# 设置随机种子以确保结果可重现
torch.manual_seed(42)

# 定义参数
seq_len = 4  # Key/Value序列长度
d_model = 6  # 特征维度

# 模拟 Key 和 Value 张量（来自编码器）
K = torch.randn(seq_len, d_model)  # 键张量
V = torch.randn(seq_len, d_model)  # 值张量

# 模拟单个 Query 向量（来自解码器当前步骤）
# 形状为 (batch_size, 1, d_model) - 每个批次一个查询
Q_single = torch.randn(1, d_model)

print("Key 张量形状:", K.shape)
print("Value 张量形状:", V.shape)
print("Query 向量形状:", Q_single.shape)

print("\n Key 值:")
print(K)

print("\n单个 Query 值:")
print(Q_single)

# 计算注意力分数 (Q和K的点积)
# 使用矩阵乘法计算Q和K的转置之间的点积
score, attention_scores = attention(Q_single, K, V)
print("\n score 值:")
print(score)
print("\n attention_scores 值:")
print(attention_scores)

