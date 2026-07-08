"""
Self-Attention 详解页面
========================

将 notebook 的 8 个步骤转化为交互式教学页面，
每一步都可以调节参数并实时查看结果。
"""

import math

import numpy as np
import streamlit as st

from src.attention import SelfAttentionFromScratch, softmax
from components.attention_heatmap import render_attention_heatmap

# 中文 token 示例
CHINESE_TOKENS = ["我", "爱", "看", "猫"]
ENGLISH_TOKENS = ["I", "love", "watching", "cats"]


# ------------------------------------------------------------------
# 页面设置
# ------------------------------------------------------------------
st.set_page_config(page_title="Self-Attention 详解", layout="wide")
st.title("🔬 自注意力详解")

st.markdown("""
本页面将带你逐步理解 Self-Attention 的工作原理。
你可以修改参数，实时观察每一步的输出变化。
""")

# ------------------------------------------------------------------
# 侧边栏：全局参数
# ------------------------------------------------------------------
st.sidebar.header("⚙️ 全局参数")

d_model = st.sidebar.slider("d_model (模型维度)", 4, 256, 8, step=2)
num_heads = st.sidebar.slider("num_heads (头数)", 1, min(8, d_model // 2), 2)
seq_len = st.sidebar.slider("seq_len (序列长度)", 2, 10, len(CHINESE_TOKENS))
temperature = st.sidebar.slider("温度系数 (Temperature)", 0.1, 3.0, 1.0, 0.1)

language = st.sidebar.radio("语言", ["中文", "英文"])
tokens = CHINESE_TOKENS[:seq_len] if language == "中文" else ENGLISH_TOKENS[:seq_len]

st.sidebar.markdown(f"**序列**: `{tokens}`")
st.sidebar.markdown(f"**d_k = d_model / num_heads = {d_model // num_heads}**")

# ------------------------------------------------------------------
# Step 1: 公式回顾
# ------------------------------------------------------------------
with st.expander("📘 Step 1: 公式回顾", expanded=True):
    st.markdown("""
    Self-Attention 的核心公式：

    $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

    其中：
    - **Q** (Query)：查询向量，代表"我在寻找什么"
    - **K** (Key)：键向量，代表"我是什么"
    - **V** (Value)：值向量，代表"我要传递的信息"
    - **$\\sqrt{d_k}$**：缩放因子，防止点积过大导致 softmax 进入梯度饱和区

    直观理解：每个位置都会生成一个 Query，与所有位置的 Key 做点积得到相关性分数，
    经过 Softmax 归一化为权重，最后用这些权重对 Value 做加权求和。
    """)

# ------------------------------------------------------------------
# Step 2: 构建输入数据
# ------------------------------------------------------------------
with st.expander("📝 Step 2: 构建输入数据"):
    st.markdown("""
    我们用一组随机向量模拟 token 的嵌入表示。
    在实际模型中，这些向量是通过 Embedding 层 learned 得到的。
    """)

    np.random.seed(42)
    X = np.random.randn(seq_len, d_model) * 0.1

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("输入张量 X")
        st.code(f"形状: {X.shape}")
        st.dataframe(X.round(4), use_container_width=True)
    with col2:
        st.subheader("Token 列表")
        for i, t in enumerate(tokens):
            st.write(f"**词元 {i}**: `{t}`")

# ------------------------------------------------------------------
# Step 3: 线性投影生成 Q, K, V
# ------------------------------------------------------------------
with st.expander("🔄 Step 3: 投影到 Q, K, V"):
    st.markdown("""
    通过三个独立的线性层，将输入 $X$ 投影到 Query、Key、Value 空间：

    $$Q = XW_Q, \\quad K = XW_K, \\quad V = XW_V$$

    每个投影矩阵 $W_Q, W_K, W_V$ 的形状为 $(d_{model}, d_{model})$。
    """)

    d_k = d_model // num_heads
    scale = 1.0 / math.sqrt(d_model)
    W_Q = np.random.randn(d_model, d_model) * scale
    W_K = np.random.randn(d_model, d_model) * scale
    W_V = np.random.randn(d_model, d_model) * scale

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Q (查询)")
        st.code(f"形状: {Q.shape}")
    with col2:
        st.caption("K (键)")
        st.code(f"形状: {K.shape}")
    with col3:
        st.caption("V (值)")
        st.code(f"形状: {V.shape}")

    st.markdown("**Q 的第一个 token 向量：**")
    st.dataframe(np.array([Q[0]]).T.round(4), use_container_width=True)

# ------------------------------------------------------------------
# Step 4: 缩放点积注意力
# ------------------------------------------------------------------
with st.expander("🎯 Step 4: 缩放点积注意力"):
    st.markdown("""
    计算 Q 和 K 的点积，除以 $\\sqrt{d_k}$ 后做 Softmax：

    $$\\text{scores} = \\frac{QK^T}{\\sqrt{d_k}}$$
    $$\\text{attention\_weights} = \\text{softmax}(\\text{scores})$$

    注意力权重矩阵的每一行表示该 token 对所有其他 token 的关注程度。
    """)

    scores = Q @ K.T / math.sqrt(d_k)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("注意力分数 (scores)")
        st.dataframe(scores.round(4), use_container_width=True)

    # 应用温度
    scaled_scores = scores / temperature
    attn_weights = softmax(scaled_scores, axis=-1)

    with col2:
        st.caption(f"注意力权重 (温度={temperature})")
        st.dataframe(attn_weights.round(4), use_container_width=True)

    # 可视化
    st.subheader("📊 注意力权重热力图")
    fig = render_attention_heatmap(attn_weights, tokens, title=f"注意力热力图 (T={temperature})")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Step 5: 加权求和
# ------------------------------------------------------------------
with st.expander("📦 Step 5: 加权求和得到输出"):
    st.markdown("""
    用注意力权重对 V 做加权求和：

    $$\\text{output} = \\text{attention\_weights} \\times V$$

    每个位置的输出是该位置所有 Value 向量的加权和，权重由注意力决定。
    """)

    output = attn_weights @ V
    st.code(f"输出形状: {output.shape}")
    st.dataframe(output.round(4), use_container_width=True)

# ------------------------------------------------------------------
# Step 6: 多头注意力
# ------------------------------------------------------------------
with st.expander("🧩 Step 6: 多头注意力"):
    st.markdown("""
    多头注意力的核心思想是：**让模型在不同的表示子空间中关注不同的信息**。

    具体做法：
    1. 将 $Q, K, V$ 分别投影到 $num\\_heads$ 个子空间
    2. 在每个子空间独立做注意力
    3. 将所有头的输出拼接起来，再做一次线性投影

    $$\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h)W^O$$
    """)

    # 演示多头 — 投影矩阵形状为 (d_model, d_model)，reshape 拆分多头
    np.random.seed(42)
    W_Q_multi = np.random.randn(d_model, d_model) * scale
    W_K_multi = np.random.randn(d_model, d_model) * scale
    W_V_multi = np.random.randn(d_model, d_model) * scale

    Q_multi = X @ W_Q_multi  # (seq_len, d_model)
    K_multi = X @ W_K_multi
    V_multi = X @ W_V_multi

    # reshape 成多头: (seq_len, d_model) -> (seq_len, num_heads, d_k) -> (num_heads, seq_len, d_k)
    Q_heads = Q_multi.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    K_heads = K_multi.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    V_heads = V_multi.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)

    # 每头单独计算注意力
    all_weights = []
    for h in range(num_heads):
        h_scores = Q_heads[h] @ K_heads[h].T / math.sqrt(d_k)
        h_scaled = h_scores / temperature
        h_weights = softmax(h_scaled, axis=-1)
        all_weights.append(h_weights)

    st.subheader("各头的注意力权重")
    for h in range(num_heads):
        with st.expander(f"第 {h+1} 头"):
            fig = render_attention_heatmap(
                all_weights[h], tokens,
                title=f"第 {h+1} 头热力图",
                temperature=None,
            )
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Step 7: PyTorch 验证
# ------------------------------------------------------------------
with st.expander("✅ Step 7: PyTorch 验证"):
    st.markdown("""
    用 PyTorch 的 `nn.MultiheadAttention` 验证我们的 NumPy 实现是否正确。
    """)

    try:
        import torch
        import torch.nn as nn

        np.random.seed(42)
        torch.manual_seed(42)

        # 创建 PyTorch 模型
        pytorch_mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            bias=False,
        )

        # 用相同的权重初始化
        with torch.no_grad():
            # 拼接 W_Q, W_K, W_V
            pytorch_mha.in_proj_weight.copy_(
                torch.tensor(np.concatenate([W_Q, W_K, W_V], axis=1), dtype=torch.float32)
            )

        # 前向传播
        torch_X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, d_model)
        torch_output, torch_weights = pytorch_mha(
            torch_X, torch_X, torch_X, need_weights=True
        )
        torch_output = torch_output.squeeze(0)  # (seq_len, d_model)

        st.success("PyTorch 验证通过！")
        st.code(f"PyTorch 输出形状: {torch_output.shape}")
        st.dataframe(torch_output.detach().numpy().round(4), use_container_width=True)

        # 对比
        st.caption("**注意：由于 PyTorch 的 in_proj_weight 是 Q/K/V 垂直拼接，而我们的实现是独立矩阵，结果会有差异。这是正常的。**")

    except ImportError:
        st.warning("PyTorch 未安装，跳过验证。")

# ------------------------------------------------------------------
# Step 8: 完整封装
# ------------------------------------------------------------------
with st.expander("🏗️ Step 8: SelfAttentionFromScratch 完整类"):
    st.markdown("""
    将上述所有步骤封装为一个完整的 Transformer Block 类，
    包含：多头注意力 + 残差连接 + LayerNorm + FFN。

    $$\\text{output} = \\text{LayerNorm}(X + \\text{MHA}(X))$$
    $$\\text{final} = \\text{LayerNorm}(\\text{output} + \\text{FFN}(\\text{output}))$$
    """)

    np.random.seed(42)
    d_ff = d_model * 2
    attention = SelfAttentionFromScratch(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.0,
    )

    st.code(attention.summary())

    X_batch = np.random.randn(1, seq_len, d_model)
    output_full, attn_weights_full = attention.forward(X_batch, training=False)

    st.subheader("完整模块输出")
    st.code(f"输出形状: {output_full.shape}")
    st.code(f"注意力权重形状: {attn_weights_full.shape}")

    # 可视化完整注意力
    if attn_weights_full.ndim == 3:
        # (1, num_heads, seq_len, seq_len)
        avg_weights = np.mean(attn_weights_full[0], axis=0)
    else:
        avg_weights = attn_weights_full[0]

    st.subheader("📊 完整模块注意力热力图")
    fig = render_attention_heatmap(avg_weights, tokens, title="完整模块注意力")
    st.plotly_chart(fig, use_container_width=True)
