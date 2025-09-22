import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现（无批次维度）

    参数:
        d_model (int): 输入和输出的特征维度
        num_heads (int): 注意力头的数量
        dropout (float, optional): Dropout比率，默认为0.1

    输入:
        Q (Tensor): 查询张量，形状为(seq_len_q, d_model)
        K (Tensor): 键张量，形状为(seq_len_k, d_model)
        V (Tensor): 值张量，形状为(seq_len_v, d_model)
        mask (Tensor, optional): 掩码张量，形状为(seq_len_q, seq_len_k)

    输出:
        output (Tensor): 注意力输出，形状为(seq_len_q, d_model)
        attention_weights (Tensor): 注意力权重，形状为(num_heads, seq_len_q, seq_len_k)
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # 查询变换
        self.W_k = nn.Linear(d_model, d_model)  # 键变换
        self.W_v = nn.Linear(d_model, d_model)  # 值变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力计算

        参数:
            Q (Tensor): 查询张量，形状为(seq_len_q, d_k)
            K (Tensor): 键张量，形状为(seq_len_k, d_k)
            V (Tensor): 值张量，形状为(seq_len_v, d_k)
            mask (Tensor, optional): 掩码张量，形状为(seq_len_q, seq_len_k)

        返回:
            output (Tensor): 注意力输出，形状为(seq_len_q, d_k)
            attention_weights (Tensor): 注意力权重，形状为(seq_len_q, seq_len_k)
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 计算输出
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    """
    前向传播

    参数:
        Q (Tensor): 查询张量，形状为(seq_len_q, d_model)
        K (Tensor): 键张量，形状为(seq_len_k, d_model)
        V (Tensor): 值张量，形状为(seq_len_v, d_model)
        mask (Tensor, optional): 掩码张量，形状为(seq_len_q, seq_len_k)

    返回:
        output (Tensor): 注意力输出，形状为(seq_len_q, d_model)
        attention_weights (Tensor): 注意力权重，形状为(num_heads, seq_len_q, seq_len_k)
    """
    def forward(self, Q, K, V, mask=None):
        # 线性变换
        Q = self.W_q(Q)  # (seq_len_q, d_model)
        K = self.W_k(K)  # (seq_len_k, d_model)
        V = self.W_v(V)  # (seq_len_v, d_model)

        # 重塑为多头形式
        Q = Q.view(-1, self.num_heads, self.d_k).transpose(0, 1)  # (num_heads, seq_len_q, d_k)
        K = K.view(-1, self.num_heads, self.d_k).transpose(0, 1)  # (num_heads, seq_len_k, d_k)
        V = V.view(-1, self.num_heads, self.d_k).transpose(0, 1)  # (num_heads, seq_len_v, d_k)

        # 如果需要，扩展掩码以匹配多头
        if mask is not None:
            mask = mask.unsqueeze(0)  # (1, seq_len_q, seq_len_k)

        # 计算缩放点积注意力
        x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 重塑回原始形状
        x = x.transpose(0, 1).contiguous().view(-1, self.d_model)  # (seq_len_q, d_model)

        # 输出线性变换
        output = self.W_o(x)

        return output, attention_weights


# 使用示例
if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)

    # 参数设置
    seq_len = 3  # 查询序列长度
    d_model = 8  # 特征维度
    num_heads = 2  # 注意力头数量

    # 创建多头注意力层
    mha = MultiHeadAttention(d_model, num_heads)

    # 创建输入张量（无批次维度）
    X = torch.randn(seq_len, d_model)
    Q = X  # 查询
    K = X  # 键
    V = X  # 值

    # 创建掩码（可选）
    # 这里创建一个简单的下三角掩码，用于自回归任务
    mask = torch.tril(torch.ones(seq_len, seq_len))  # (seq_len_q, seq_len_kv)

    print("输入形状:")
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    print(f"mask: {mask.shape}")

    # 前向传播
    output, attention_weights = mha.forward(Q, K, V)

    print("\n输出形状:")
    print(f"output: {output.shape}")
    print(f"attention_weights: {attention_weights.shape}")

    # 验证输出
    print("\n验证:")
    print(f"输入和输出维度是否一致: {Q.shape == output.shape}")

    # 显示一个注意力头的权重
    print(f"\n第一个头的注意力权重:")
    print(attention_weights[0])