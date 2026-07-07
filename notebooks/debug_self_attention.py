"""
手撕 Self-Attention 机制
========================
包含两个版本：
1. 纯 NumPy 实现单头自注意力（面试手撕常考版本，逻辑最清晰）
2. PyTorch 实现多头自注意力（工程实践常用版本）
"""

import numpy as np


# ============================================================
# 版本一：纯 NumPy 手撕单头 Self-Attention
# ============================================================
def softmax(x, axis=-1):
    # 减去最大值防止指数爆炸（数值稳定性技巧，面试常问）
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class SelfAttentionNumpy:
    def __init__(self, d_model, d_k):
        """
        d_model: 输入向量维度
        d_k: Q/K/V 投影后的维度
        """
        # 随机初始化权重矩阵 W_q, W_k, W_v
        self.W_q = np.random.randn(d_model, d_k) * 0.01
        self.W_k = np.random.randn(d_model, d_k) * 0.01
        self.W_v = np.random.randn(d_model, d_k) * 0.01
        self.d_k = d_k

    def forward(self, X, mask=None):
        """
        X: (seq_len, d_model) 输入序列
        mask: (seq_len, seq_len) 可选，用于遮蔽未来信息(如decoder) 或 padding
        """
        Q = X @ self.W_q       # (seq_len, d_k)
        K = X @ self.W_k       # (seq_len, d_k)
        V = X @ self.W_v       # (seq_len, d_k)

        # 1. 计算注意力分数：Q 与 K 的点积
        scores = Q @ K.T / np.sqrt(self.d_k)   # (seq_len, seq_len)

        # 2. 可选：加 mask（把要屏蔽的位置设为 -inf，softmax后趋近于0）
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # 3. softmax 归一化，得到注意力权重
        attn_weights = softmax(scores, axis=-1)   # (seq_len, seq_len)

        # 4. 加权求和 V，得到输出
        output = attn_weights @ V   # (seq_len, d_k)

        return output, attn_weights


# ---------------- 测试 NumPy 版本 ----------------
if __name__ == "__main__":
    np.random.seed(0)
    seq_len, d_model, d_k = 4, 8, 6
    X = np.random.randn(seq_len, d_model)

    attn = SelfAttentionNumpy(d_model, d_k)
    out, weights = attn.forward(X)

    print("=== NumPy 单头 Self-Attention ===")
    print("输出 shape:", out.shape)          # (4, 6)
    print("注意力权重 shape:", weights.shape)  # (4, 4)
    print("每行权重和应为1:", weights.sum(axis=-1))


# ============================================================
# 版本二：PyTorch 手撕多头 Self-Attention（Multi-Head Attention）
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # 每个头的维度

        # 一次性投影出 Q, K, V（比分开写三个Linear更省参数、更常见）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)   # 多头拼接后的输出投影

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, 1, seq_len, seq_len) 或可广播的形状
        """
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x)   # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 拆分成多头: (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)   # (batch, num_heads, seq_len, d_k)

        # 合并多头: (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 输出投影
        output = self.W_o(attn_output)
        return output, attn_weights


# ---------------- 测试 PyTorch 版本 ----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_model, num_heads = 2, 5, 16, 4
    x = torch.randn(batch, seq_len, d_model)

    mha = MultiHeadSelfAttention(d_model, num_heads)
    out, attn = mha(x)

    print("\n=== PyTorch 多头 Self-Attention ===")
    print("输出 shape:", out.shape)   # (2, 5, 16)
    print("注意力权重 shape:", attn.shape)  # (2, 4, 5, 5)

    # 演示causal mask（decoder用，防止看到未来token）
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    out_masked, attn_masked = mha(x, mask=causal_mask)
    print("\nCausal mask 下第一行的注意力权重（只能看到自己）:")
    print(attn_masked[0, 0, 0])
