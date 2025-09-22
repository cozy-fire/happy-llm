import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
串行计算的多头注意力机制实现

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
    all_attention_weights (list): 所有注意力头的权重列表
"""
class SequentialMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(SequentialMultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 为每个头创建独立的线性变换层
        self.W_q_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_k_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])
        self.W_v_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(num_heads)])

        # 输出线性变换层
        self.W_o = nn.Linear(d_model, d_model)

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
    前向传播 - 串行计算每个注意力头

    参数:
        Q (Tensor): 查询张量，形状为(seq_len_q, d_model)
        K (Tensor): 键张量，形状为(seq_len_k, d_model)
        V (Tensor): 值张量，形状为(seq_len_v, d_model)
        mask (Tensor, optional): 掩码张量，形状为(seq_len_q, seq_len_k)

    返回:
        output (Tensor): 注意力输出，形状为(seq_len_q, d_model)
        all_attention_weights (list): 所有注意力头的权重列表
    """
    def forward(self, Q, K, V, mask=None):

        # 存储所有头的输出
        head_outputs = []
        all_attention_weights = []

        # 对每个头进行串行计算
        for i in range(self.num_heads):
            # 获取当前头的线性变换层
            W_q = self.W_q_layers[i]
            W_k = self.W_k_layers[i]
            W_v = self.W_v_layers[i]

            # 对当前头进行线性变换
            Q_head = W_q(Q)  # (seq_len_q, d_k)
            K_head = W_k(K)  # (seq_len_k, d_k)
            V_head = W_v(V)  # (seq_len_v, d_k)

            # 计算当前头的注意力
            head_output, attention_weights = self.scaled_dot_product_attention(
                Q_head, K_head, V_head, mask
            )

            # 保存当前头的输出和注意力权重
            head_outputs.append(head_output)
            all_attention_weights.append(attention_weights)

            # 打印当前头的计算信息（用于理解）
            print(f"头 {i + 1}:")
            print(f"  Q_head 形状: {Q_head.shape}")
            print(f"  K_head 形状: {K_head.shape}")
            print(f"  V_head 形状: {V_head.shape}")
            print(f"  注意力权重形状: {attention_weights.shape}")
            print(f"  头输出形状: {head_output.shape}")
            print()

        # 将所有头的输出拼接起来
        concatenated = torch.cat(head_outputs, dim=-1)  # (seq_len_q, d_model)

        # 输出线性变换
        output = self.W_o(concatenated)

        return output, all_attention_weights


# 使用示例
if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)

    # 参数设置
    seq_len = 3  # 查询序列长度
    d_model = 8  # 特征维度
    num_heads = 2  # 注意力头数量

    # 创建串行多头注意力层
    smha = SequentialMultiHeadAttention(d_model, num_heads)

    # 创建输入张量（无批次维度）
    X = torch.randn(seq_len, d_model)
    Q = X # 查询
    K = X  # 键
    V = X  # 值

    print("输入形状:")
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    print()

    # 前向传播
    output, all_attention_weights = smha.forward(Q, K, V)

    print("\n最终输出形状:")
    print(f"output: {output.shape}")

    # 验证输出
    print("\n验证:")
    print(f"输入和输出维度是否一致: {Q.shape == output.shape}")

    # 显示每个头的注意力权重
    print(f"\n各头的注意力权重:")
    for i, weights in enumerate(all_attention_weights):
        print(f"头 {i + 1}:")
        print(weights)
        print()