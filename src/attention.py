"""
从零实现的 Self-Attention 模块
================================

本模块包含两个版本的 Self-Attention 实现：

1. **NumPy 版本** — 纯矩阵运算，用于教学和理解原理
2. **PyTorch 版本** — nn.Module 封装，用于工程实践和验证

包含组件：
- Scaled Dot-Product Attention
- Multi-Head Attention
- Self-Attention From Scratch（含残差连接 + LayerNorm + FFN）
- 手动反向传播梯度推导

参考资料：
- "Attention Is All You Need" (Vaswani et al., 2017)
- 对应 notebook: notebooks/01-self-attention-from-scratch.ipynb
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import numpy as np


# ====================================================================
# 工具函数
# ====================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    数值稳定的 softmax。

    减去最大值防止 exp 溢出，这是面试常问的细节。

    Args:
        x: 输入数组
        axis: 沿哪个轴做 softmax

    Returns:
        softmax 后的数组，沿 axis 方向和为 1
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def causal_mask(seq_len: int) -> np.ndarray:
    """
    生成上三角因果掩码（decoder 用，防止看到未来 token）。

    Args:
        seq_len: 序列长度

    Returns:
        (seq_len, seq_len) 布尔矩阵，上三角为 False（遮蔽）
    """
    mask = np.tril(np.ones((seq_len, seq_len))).astype(bool)
    return mask


# ====================================================================
# 版本一：纯 NumPy 单头 Self-Attention
# ====================================================================

class SelfAttentionNumpy:
    """
    纯 NumPy 实现的单头自注意力机制。

    这是面试手撕代码的经典版本，逻辑最清晰，适合理解每一步的矩阵运算。

    Attributes:
        W_q: (d_model, d_k) 查询投影权重
        W_k: (d_model, d_k) 键投影权重
        W_v: (d_model, d_k) 值投影权重
        d_k: 每个头的维度
    """

    def __init__(self, d_model: int, d_k: int):
        scale = 1.0 / math.sqrt(d_model)
        self.W_q = np.random.randn(d_model, d_k) * scale
        self.W_k = np.random.randn(d_model, d_k) * scale
        self.W_v = np.random.randn(d_model, d_k) * scale
        self.d_k = d_k

    def forward(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播。

        Args:
            X: (seq_len, d_model) 输入序列
            mask: (seq_len, seq_len) 可选掩码，0 表示遮蔽

        Returns:
            output: (seq_len, d_k) 注意力输出
            attn_weights: (seq_len, seq_len) 注意力权重
        """
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # 注意力分数 = Q @ K^T / sqrt(d_k)
        scores = Q @ K.T / math.sqrt(self.d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax 归一化
        attn_weights = softmax(scores, axis=-1)

        # 加权求和
        output = attn_weights @ V

        return output, attn_weights


# ====================================================================
# 版本二：多头注意力核心函数（NumPy）
# ====================================================================

def split_heads(
    x: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """
    将 (batch, seq_len, d_model) 重塑为 (batch, num_heads, seq_len, d_k)。

    操作：先 reshape 再 transpose。

    Args:
        x: 输入张量
        num_heads: 头数

    Returns:
        重塑后的张量
    """
    batch, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    return x.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)


def merge_heads(
    x: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """
    将 (batch, num_heads, seq_len, d_k) 重塑回 (batch, seq_len, d_model)。

    Args:
        x: 多头张量
        num_heads: 头数

    Returns:
        合并后的张量
    """
    batch, num_heads, seq_len, d_k = x.shape
    d_model = num_heads * d_k
    return x.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    dropout_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    缩放点积注意力核心计算。

    $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

    Args:
        Q: (batch, num_heads, seq_len, d_k) 查询
        K: (batch, num_heads, seq_len, d_k) 键
        V: (batch, num_heads, seq_len, d_k) 值
        dropout_mask: 可选的 dropout 掩码

    Returns:
        output: (batch, num_heads, seq_len, d_k)
        attn_weights: (batch, num_heads, seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # Q @ K^T -> (batch, num_heads, seq_len, seq_len)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(d_k)

    # Softmax（数值稳定：减最大值）
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Dropout（训练时）
    if dropout_mask is not None:
        attn_weights = attn_weights * dropout_mask
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    # 加权求和
    output = np.matmul(attn_weights, V)

    return output, attn_weights


def multi_head_attention_numpy(
    X: np.ndarray,
    num_heads: int,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W_O: np.ndarray,
    dropout_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    完整的多头注意力前向传播（NumPy 版）。

    Args:
        X: (batch, seq_len, d_model) 输入
        num_heads: 头数
        W_Q, W_K, W_V: (d_model, d_model) 投影权重
        W_O: (d_model, d_model) 输出投影权重
        dropout_mask: 可选 dropout 掩码

    Returns:
        output: (batch, seq_len, d_model)
        attn_weights: (batch, num_heads, seq_len, seq_len)
    """
    batch_size, _, d_model = X.shape

    # 线性投影
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # 拆分多头
    Q_heads = split_heads(Q, num_heads)
    K_heads = split_heads(K, num_heads)
    V_heads = split_heads(V, num_heads)

    # 注意力计算
    attn_output_per_head, attn_weights = scaled_dot_product_attention(
        Q_heads, K_heads, V_heads, dropout_mask
    )

    # 拼接 + 输出投影
    concat_output = merge_heads(attn_output_per_head, num_heads)
    output = concat_output @ W_O

    return output, attn_weights


# ====================================================================
# 版本三：手动反向传播
# ====================================================================

def scaled_dot_product_attention_backward(
    d_output: np.ndarray,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    attn_weights: np.ndarray,
    d_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention 的手动反向传播。

    前向: attn = softmax(QK^T / sqrt(d_k)) @ V

    梯度推导:
        1. dV = attn_weights^T @ d_output
        2. d_attn = d_output @ V^T
        3. d_scores = attn * (d_attn - sum(d_attn * attn, axis=-1))
        4. d_QK = d_scores / sqrt(d_k)
        5. dQ = d_QK @ K^T, dK = d_QK^T @ Q

    Args:
        d_output: (batch, num_heads, seq_len, d_k) 上游梯度
        Q, K, V: 前向的 Q, K, V
        attn_weights: 前向的注意力权重
        d_k: 每个头的维度

    Returns:
        dQ, dK, dV
    """
    # dV
    dV = attn_weights.transpose(0, 1, 3, 2) @ d_output

    # d_attn
    d_attn = d_output @ V.transpose(0, 1, 3, 2)

    # softmax 导数
    d_scaled_scores = attn_weights * (
        d_attn - np.sum(d_attn * attn_weights, axis=-1, keepdims=True)
    )

    # 缩放逆操作
    d_QK = d_scaled_scores / math.sqrt(d_k)

    # Q @ K^T 的梯度
    dQ = d_QK @ K.transpose(0, 1, 3, 2)
    dK = d_QK.transpose(0, 1, 3, 2) @ Q

    return dQ, dK, dV


# ====================================================================
# 版本四：完整封装类（含残差 + LayerNorm + FFN）
# ====================================================================

class SelfAttentionFromScratch:
    """
    从头实现的 Self-Attention 模块（完整 Transformer Block）。

    包含：
    - Multi-Head Attention
    - Layer Normalization
    - Position-wise Feed Forward Network (ReLU)
    - 残差连接

    结构：
        h = LayerNorm(X + MHA(X))
        output = LayerNorm(h + FFN(h))

    参考 notebook: notebooks/01-self-attention-from-scratch.ipynb Step 8

    Attributes:
        d_model: 模型维度
        num_heads: 注意力头数
        d_k: 每个头的维度
        dropout_rate: Dropout 比例
        W_Q, W_K, W_V, W_O: 注意力权重
        W_1, b_1, W_2, b_2: FFN 权重
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout

        # Xavier 初始化
        scale = 1.0 / math.sqrt(d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

        # FFN 权重
        self.W_1 = np.random.randn(d_model, d_ff) * scale
        self.b_1 = np.zeros(d_ff)
        self.W_2 = np.random.randn(d_ff, d_model) * scale
        self.b_2 = np.zeros(d_model)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.num_heads, self.d_k).transpose(
            0, 2, 1, 3
        )

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        batch, num_heads, seq_len, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(
            batch, seq_len, num_heads * d_k
        )

    def _scaled_dot_product_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        training: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)

        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        if training and self.dropout_rate > 0:
            mask = (
                np.random.rand(*attn_weights.shape) > self.dropout_rate
            ).astype(np.float32)
            attn_weights = attn_weights * mask / (1 - self.dropout_rate)

        output = np.matmul(attn_weights, V)
        return output, attn_weights

    @staticmethod
    def _layer_norm(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        Layer Normalization：对最后一个维度做归一化。

        $$\\text{LayerNorm}(x) = \\frac{x - \\mu}{\\sigma + \\epsilon}$$
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + epsilon)

    def _ff(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Position-wise Feed Forward Network (ReLU 激活)。

        $$\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2$$
        """
        h = np.maximum(0, x @ self.W_1 + self.b_1)
        if training and self.dropout_rate > 0:
            mask = (np.random.rand(*h.shape) > self.dropout_rate).astype(
                np.float32
            )
            h = h * mask / (1 - self.dropout_rate)
        return h @ self.W_2 + self.b_2

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def forward(
        self,
        X: np.ndarray,
        training: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        完整的前向传播（带残差连接 + LayerNorm）。

        Args:
            X: (batch, seq_len, d_model) 输入
            training: 是否训练模式（启用 dropout）

        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, num_heads, seq_len, seq_len)
        """
        # --- Sublayer 1: Multi-Head Attention ---
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        attn_output, attn_weights = self._scaled_dot_product_attention(
            Q_heads, K_heads, V_heads, training=training
        )

        attn_concat = self._merge_heads(attn_output)
        attn_out = attn_concat @ self.W_O

        # 残差 + LayerNorm
        h1 = self._layer_norm(X + attn_out)

        # --- Sublayer 2: Feed Forward ---
        ff_out = self._ff(h1, training=training)

        # 残差 + LayerNorm
        output = self._layer_norm(h1 + ff_out)

        return output, attn_weights

    def summary(self) -> str:
        """打印模块配置信息和参数量。"""
        total_params = (
            self.d_model**2 * 4  # W_Q, W_K, W_V, W_O
            + self.d_model * self.d_ff * 2  # W_1, W_2
            + self.d_ff + self.d_model  # b_1, b_2
        )
        return (
            f"Self-Attention 模块配置:\n"
            f"  d_model:    {self.d_model}\n"
            f"  num_heads:  {self.num_heads}\n"
            f"  d_k:        {self.d_k}\n"
            f"  d_ff:       {self.d_ff}\n"
            f"  dropout:    {self.dropout_rate}\n"
            f"  参数量:     {total_params:,}"
        )


# ====================================================================
# 测试入口
# ====================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # ---- 测试 1: NumPy 单头 ----
    print("=" * 50)
    print("测试 1: NumPy 单头 Self-Attention")
    print("=" * 50)
    seq_len, d_model, d_k = 4, 8, 6
    X = np.random.randn(seq_len, d_model)
    attn = SelfAttentionNumpy(d_model, d_k)
    out, weights = attn.forward(X)
    print(f"输出 shape: {out.shape}")
    print(f"注意力权重 shape: {weights.shape}")
    print(f"每行权重和: {weights.sum(axis=-1)}")

    # ---- 测试 2: 完整类 ----
    print()
    print("=" * 50)
    print("测试 2: SelfAttentionFromScratch 完整类")
    print("=" * 50)
    attention = SelfAttentionFromScratch(d_model=6, num_heads=3, d_ff=12)
    print(attention.summary())

    X_batch = np.random.randn(1, 4, 6)
    output, weights = attention.forward(X_batch, training=False)
    print(f"\n输出 shape: {output.shape}")
    print(f"注意力权重 shape: {weights.shape}")

    # ---- 测试 3: PyTorch 对比验证 ----
    print()
    print("=" * 50)
    print("测试 3: PyTorch 对比验证")
    print("=" * 50)
    try:
        import torch
        import torch.nn as nn

        torch_X = torch.tensor(X_batch, dtype=torch.float32)
        torch_W_Q = torch.tensor(attention.W_Q, dtype=torch.float32)
        torch_W_K = torch.tensor(attention.W_K, dtype=torch.float32)
        torch_W_V = torch.tensor(attention.W_V, dtype=torch.float32)
        torch_W_O = torch.tensor(attention.W_O, dtype=torch.float32)

        mha_torch = nn.MultiheadAttention(
            embed_dim=6, num_heads=3, batch_first=True
        )
        with torch.no_grad():
            mha_torch.in_proj_weight.copy_(
                torch.cat([torch_W_Q, torch_W_K, torch_W_V])
            )
            mha_torch.out_proj.weight.copy_(torch_W_O)

        torch_output, _ = mha_torch(torch_X, torch_X, torch_X)
        diff = np.abs(output - torch_output.detach().numpy()).max()
        print(f"NumPy vs PyTorch 最大差异: {diff:.6e}")
        if diff < 1e-5:
            print("✓ 两者结果一致！")
        else:
            print("⚠ 存在差异（可能是权重初始化方式不同）")
    except ImportError:
        print("PyTorch 未安装，跳过验证。")
