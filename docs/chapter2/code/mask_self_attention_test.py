import math
import torch
from torch.onnx.symbolic_opset9 import tensor
import torch.nn.functional as F

def mask_attention(query, key, value):
    '''
    args:
    query: 查询值矩阵
    key: 键值矩阵
    value: 真值矩阵
    '''
    # 创建Mask矩阵
    vec_len = query.size(0)
    mask = torch.full((vec_len, vec_len), float("-inf"))
    # triu 函数的功能是创建一个上三角矩阵
    mask = torch.triu(mask, diagonal=1)
    print("mask矩阵:")
    print(mask)
    # 获取键向量的维度，键向量的维度和值向量的维度相同
    d_k = query.size(-1)
    # 计算Q与K的内积并除以根号dk
    # transpose——相当于转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    print("\n初始权重矩阵score：")
    print(scores)
    # 计算结果与mask矩阵相加
    scores = mask + scores
    print("\nscore + mask：")
    print(scores)
    # Softmax
    p_attn = scores.softmax(dim=-1)
    print("\nsoftmax归一化后：")
    print(p_attn)
     # 根据计算结果对value进行加权求和
    return torch.matmul(p_attn, value), p_attn

# 设置随机种子以确保结果可重现
torch.manual_seed(42)

# 创建一个上三角矩阵，用于遮蔽未来信息。
# 先通过 full 函数创建一个 1 * seq_len * seq_len 的矩阵
max_seq_len = 5

# 定义参数
seq_len = 4  # Key/Value序列长度
d_model = 6  # 特征维度

# 模拟 数据源 X
X = torch.randn(seq_len, d_model)

# 模拟Query Key 和 Value 张量，自注意力机制中，都来自相同数据源X(省略了权重转换过程)
Q = X  # 查询对象张量
K = X  # 键张量
V = X  # 值张量

# 模拟单个 Query 向量，从

print("\nKey 张量形状:", K.shape)
print("Value 张量形状:", V.shape)
print("Query 张量形状:", Q.shape)

print("\n Key 值:")
print(K)

print("\nQuery 值:")
print(Q)

# 计算注意力分数 (Q和K的点积)
# 使用矩阵乘法计算Q和K的转置之间的点积
Z, attention_scores = mask_attention(Q, K, V)
print("\n Z 值:")
print(Z)

