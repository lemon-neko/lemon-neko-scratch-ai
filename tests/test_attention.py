"""
Self-Attention 单元测试
========================

覆盖 src/attention.py 和 src/layers.py 中的核心功能。
"""

import math

import numpy as np
import pytest

# 确保 src 在路径中
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.attention import (
    SelfAttentionFromScratch,
    SelfAttentionNumpy,
    causal_mask,
    multi_head_attention_numpy,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
    softmax,
    split_heads,
    merge_heads,
)
from src.layers import (
    causal_mask as layers_causal_mask,
    feed_forward,
    layer_norm,
    padding_mask,
    positional_encoding,
)


# ====================================================================
# softmax 测试
# ====================================================================

class TestSoftmax:
    def test_basic(self):
        """softmax 输出和应为 1"""
        x = np.array([[1.0, 2.0, 3.0]])
        result = softmax(x, axis=-1)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_numerical_stability(self):
        """大数值不应溢出"""
        x = np.array([[1e10, 1e10 + 1, 1e10 + 2]])
        result = softmax(x, axis=-1)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_uniform(self):
        """均匀输入应产生均匀输出"""
        x = np.array([[0.0, 0.0, 0.0]])
        result = softmax(x, axis=-1)
        expected = np.array([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
        np.testing.assert_allclose(result, expected, atol=1e-6)


# ====================================================================
# causal_mask 测试
# ====================================================================

class TestCausalMask:
    def test_shape(self):
        mask = causal_mask(5)
        assert mask.shape == (5, 5)

    def test_upper_triangle_zeroed(self):
        mask = causal_mask(4)
        # 上三角应为 False
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j] is False
        # 对角线及下三角应为 True
        for i in range(4):
            for j in range(i + 1):
                assert mask[i, j] is True

    def test_identity_on_diagonal(self):
        mask = causal_mask(3)
        np.testing.assert_array_equal(mask.diagonal(), np.ones(3, dtype=bool))


# ====================================================================
# SelfAttentionNumpy 测试
# ====================================================================

class TestSelfAttentionNumpy:
    def test_output_shape(self):
        attn = SelfAttentionNumpy(d_model=8, d_k=4)
        X = np.random.randn(5, 8)
        output, weights = attn.forward(X)
        assert output.shape == (5, 4)
        assert weights.shape == (5, 5)

    def test_mask_applied(self):
        attn = SelfAttentionNumpy(d_model=8, d_k=4)
        X = np.random.randn(3, 8)
        mask = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        output, weights = attn.forward(X, mask=mask)
        # 被 mask 遮蔽的位置权重应为 0
        assert np.allclose(weights[mask == 0], 0.0, atol=1e-9)


# ====================================================================
# split_heads / merge_heads 测试
# ====================================================================

class TestHeadOperations:
    def test_split_merge_roundtrip(self):
        batch, seq_len, d_model, num_heads = 2, 5, 16, 4
        x = np.random.randn(batch, seq_len, d_model)
        d_k = d_model // num_heads

        split = split_heads(x, num_heads)
        assert split.shape == (batch, num_heads, seq_len, d_k)

        merged = merge_heads(split, num_heads)
        np.testing.assert_array_almost_equal(merged, x)


# ====================================================================
# scaled_dot_product_attention 测试
# ====================================================================

class TestScaledDotProductAttention:
    def test_output_shape(self):
        batch, num_heads, seq_len, d_k = 2, 4, 5, 8
        Q = np.random.randn(batch, num_heads, seq_len, d_k)
        K = np.random.randn(batch, num_heads, seq_len, d_k)
        V = np.random.randn(batch, num_heads, seq_len, d_k)

        output, weights = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (batch, num_heads, seq_len, d_k)
        assert weights.shape == (batch, num_heads, seq_len, seq_len)

    def test_attention_sum_to_one(self):
        batch, num_heads, seq_len, d_k = 1, 2, 3, 4
        Q = np.random.randn(batch, num_heads, seq_len, d_k)
        K = np.random.randn(batch, num_heads, seq_len, d_k)
        V = np.random.randn(batch, num_heads, seq_len, d_k)

        _, weights = scaled_dot_product_attention(Q, K, V)
        # 每行的注意力权重和应为 1
        for b in range(batch):
            for h in range(num_heads):
                row_sums = weights[b, h].sum(axis=-1)
                np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ====================================================================
# SelfAttentionFromScratch 测试
# ====================================================================

class TestSelfAttentionFromScratch:
    def test_output_shape(self):
        model = SelfAttentionFromScratch(d_model=16, num_heads=4, d_ff=32)
        X = np.random.randn(2, 5, 16)
        output, weights = model.forward(X, training=False)
        assert output.shape == (2, 5, 16)
        assert weights.shape == (2, 4, 5, 5)

    def test_summary_contains_config(self):
        model = SelfAttentionFromScratch(d_model=64, num_heads=8, d_ff=128)
        summary = model.summary()
        assert "d_model" in summary
        assert "64" in summary
        assert "8" in summary
        assert "128" in summary

    def test_training_vs_inference(self):
        model = SelfAttentionFromScratch(d_model=16, num_heads=4, d_ff=32, dropout=0.5)
        X = np.random.randn(1, 3, 16)

        np.random.seed(42)
        out_train, _ = model.forward(X, training=True)

        np.random.seed(42)
        out_eval, _ = model.forward(X, training=False)

        # 训练模式下 dropout 会导致输出不同
        assert not np.allclose(out_train, out_eval, atol=1e-6)


# ====================================================================
# layer_norm 测试
# ====================================================================

class TestLayerNorm:
    def test_normalization(self):
        x = np.random.randn(2, 3, 8)
        normalized, mean, std = layer_norm(x)
        # 最后一维的均值应接近 0
        np.testing.assert_allclose(normalized.mean(axis=-1), 0.0, atol=1e-5)
        # 最后一维的标准差应接近 1
        np.testing.assert_allclose(normalized.std(axis=-1), 1.0, atol=1e-5)

    def test_with_gamma_beta(self):
        x = np.random.randn(2, 3, 8)
        gamma = np.ones(8)
        beta = np.zeros(8)
        normalized, _, _ = layer_norm(x, gamma=gamma, beta=beta)
        assert normalized.shape == x.shape


# ====================================================================
# feed_forward 测试
# ====================================================================

class TestFeedForward:
    def test_output_shape(self):
        x = np.random.randn(2, 5, 16)
        W_1 = np.random.randn(16, 32)
        b_1 = np.zeros(32)
        W_2 = np.random.randn(32, 16)
        b_2 = np.zeros(16)

        output = feed_forward(x, W_1, b_1, W_2, b_2)
        assert output.shape == (2, 5, 16)

    def test_relu_activation(self):
        x = np.array([[[-1.0, -2.0, 3.0]]])
        W_1 = np.ones((3, 3))
        b_1 = np.zeros(3)
        W_2 = np.ones((3, 3))
        b_2 = np.zeros(3)

        output = feed_forward(x, W_1, b_1, W_2, b_2)
        # ReLU 应将负值归零
        assert np.all(output >= 0)


# ====================================================================
# positional_encoding 测试
# ====================================================================

class TestPositionalEncoding:
    def test_shape(self):
        pe = positional_encoding(seq_len=10, d_model=16)
        assert pe.shape == (10, 16)

    def test_values_bounded(self):
        pe = positional_encoding(seq_len=20, d_model=32)
        assert np.all(np.abs(pe) <= 1.0)

    def test_different_positions(self):
        pe = positional_encoding(seq_len=5, d_model=8)
        # 不同位置的编码应不同
        assert not np.allclose(pe[0], pe[1])


# ====================================================================
# padding_mask 测试
# ====================================================================

class TestPaddingMask:
    def test_shape(self):
        mask = padding_mask(5, 3)
        assert mask.shape == (1, 1, 5, 3)

    def test_all_ones(self):
        mask = padding_mask(4, 6)
        np.testing.assert_array_almost_equal(mask, np.ones((1, 1, 4, 6)))


# ====================================================================
# 集成测试
# ====================================================================

class TestIntegration:
    def test_full_attention_pipeline(self):
        """端到端测试：输入 → 注意力 → 输出"""
        np.random.seed(42)
        model = SelfAttentionFromScratch(d_model=16, num_heads=4, d_ff=32)
        X = np.random.randn(2, 5, 16)

        output, weights = model.forward(X, training=False)

        # 输出形状正确
        assert output.shape == (2, 5, 16)
        # 注意力权重合法
        assert weights.shape == (2, 4, 5, 5)
        # 每行权重和为 1
        for b in range(2):
            for h in range(4):
                row_sums = weights[b, h].sum(axis=-1)
                np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
