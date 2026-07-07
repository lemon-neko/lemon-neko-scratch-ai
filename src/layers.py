"""
基础神经网络层（从零实现）
============================

本模块包含 Transformer 架构中用到的基础构建块：

- Layer Normalization
- Position-wise Feed Forward Network
- Sinusoidal Positional Encoding
- Causal Mask 生成

所有实现均为纯 NumPy，不依赖任何深度学习框架。

参考：
- "Attention Is All You Need" (Vaswani et al., 2017) — Section 3.4 (Positional Encoding)
- 对应 notebook: notebooks/01-self-attention-from-scratch.ipynb
"""

from __future__ import annotations

from typing import Optional, Tuple

import math

import numpy as np


# ====================================================================
# Layer Normalization
# ====================================================================

def layer_norm(
    x: np.ndarray,
    epsilon: float = 1e-6,
    gamma: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Layer Normalization：对最后一个维度做归一化。

    $$\\text{LayerNorm}(x) = \\gamma \\cdot \\frac{x - \\mu}{\\sigma + \\epsilon} + \\beta$$

    与 BatchNorm 不同，LayerNorm 是在单个样本的特征维度上做归一化，
    不受 batch size 影响，更适合 Transformer。

    Args:
        x: 输入张量，形状任意，最后一个维度为特征维度
        epsilon: 数值稳定性常数
        gamma: 可学习的缩放参数（可选）
        beta: 可学习的偏移参数（可选）

    Returns:
        normalized: 归一化后的张量
        mean: 均值（用于反向传播）
        std: 标准差（用于反向传播）
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    normalized = (x - mean) / (std + epsilon)

    if gamma is not None:
        normalized = normalized * gamma
    if beta is not None:
        normalized = normalized + beta

    return normalized, mean, std


# ====================================================================
# Position-wise Feed Forward Network
# ====================================================================

def feed_forward(
    x: np.ndarray,
    W_1: np.ndarray,
    b_1: np.ndarray,
    W_2: np.ndarray,
    b_2: np.ndarray,
    training: bool = False,
    dropout_rate: float = 0.0,
) -> np.ndarray:
    """
    Position-wise Feed Forward Network（ReLU 激活）。

    $$\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2$$

    在 Transformer 中，FFN 对每个位置独立应用，故称 "position-wise"。
    原始论文使用 ReLU，后续工作也常用 GELU。

    Args:
        x: (..., d_model) 输入
        W_1: (d_model, d_ff) 第一层权重
        b_1: (d_ff,) 第一层偏置
        W_2: (d_ff, d_model) 第二层权重
        b_2: (d_model,) 第二层偏置
        training: 是否训练模式
        dropout_rate: Dropout 比例

    Returns:
        output: (..., d_model)
    """
    h = np.maximum(0, x @ W_1 + b_1)

    if training and dropout_rate > 0:
        mask = (np.random.rand(*h.shape) > dropout_rate).astype(np.float32)
        h = h * mask / (1 - dropout_rate)

    return h @ W_2 + b_2


# ====================================================================
# Sinusoidal Positional Encoding
# ====================================================================

def positional_encoding(
    seq_len: int,
    d_model: int,
    max_len: int = 5000,
) -> np.ndarray:
    """
    正弦位置编码。

    $$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$
    $$PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$

    位置编码的作用是让模型知道每个 token 在序列中的位置。
    由于 Self-Attention 本身是排列不变的（permutation invariant），
    必须通过某种方式注入位置信息。

    Args:
        seq_len: 序列长度
        d_model: 模型维度
        max_len: 最大序列长度（控制频率范围）

    Returns:
        (seq_len, d_model) 位置编码矩阵
    """
    pos = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(
        np.arange(0, d_model, 2) * (-math.log(max_len) / d_model)
    )  # (d_model // 2,)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)

    return pe


# ====================================================================
# Causal Mask
# ====================================================================

def causal_mask(seq_len: int) -> np.ndarray:
    """
    生成上三角因果掩码（decoder 用）。

    防止 decoder 在训练/推理时看到未来的 token。

    Args:
        seq_len: 序列长度

    Returns:
        (seq_len, seq_len) 布尔矩阵，下三角为 True（可见），上三角为 False（遮蔽）
    """
    return np.tril(np.ones((seq_len, seq_len))).astype(bool)


def padding_mask(seq_len_k: int, seq_len_v: int) -> np.ndarray:
    """
    生成 padding 掩码。

    用于处理变长序列的 padding token。

    Args:
        seq_len_k: key 序列长度
        seq_len_v: value 序列长度

    Returns:
        (1, 1, seq_len_k, seq_len_v) 掩码张量
    """
    mask = np.ones((1, 1, seq_len_k, seq_len_v))
    return mask.astype(np.float32)
