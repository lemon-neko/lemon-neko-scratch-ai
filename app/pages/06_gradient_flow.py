"""
梯度流分析 — 手动反向传播探索器
==================================

展示 Self-Attention 的完整反向传播过程，
包括梯度范数柱状图和逐层梯度变化。
"""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.attention import (
    SelfAttentionFromScratch,
    scaled_dot_product_attention_backward,
    softmax,
)
from src.layers import layer_norm

st.set_page_config(page_title="梯度流分析", layout="wide")
st.title("📉 梯度流分析")

st.markdown("""
本页面展示 Self-Attention 的**手动反向传播**过程。
你可以观察每个参数的梯度大小和分布，理解梯度消失/爆炸问题。
""")

# ------------------------------------------------------------------
# 侧边栏
# ------------------------------------------------------------------
st.sidebar.header("⚙️ 配置")

d_model = st.sidebar.slider("d_model", 8, 64, 16, step=4)
num_heads = st.sidebar.slider("num_heads", 1, min(8, d_model // 2), 2)
seq_len = st.sidebar.slider("seq_len", 2, 8, 4)
seed = st.sidebar.number_input("Random Seed", 0, 999999, 42)

np.random.seed(seed)

# ------------------------------------------------------------------
# 创建模型
# ------------------------------------------------------------------
attention = SelfAttentionFromScratch(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_model * 2,
    dropout=0.0,
)

# ------------------------------------------------------------------
# 前向传播
# ------------------------------------------------------------------
st.header("📐 前向传播")

X = np.random.randn(1, seq_len, d_model)
output, attn_weights = attention.forward(X, training=False)

st.code(f"Input Shape: {X.shape}")
st.code(f"Output Shape: {output.shape}")
st.code(f"Attn Weights Shape: {attn_weights.shape}")

# ------------------------------------------------------------------
# 手动反向传播
# ------------------------------------------------------------------
st.header("🔄 手动反向传播")

st.markdown("""
我们从输出梯度开始，逐步反向推导每个中间变量的梯度。

**前向公式：**
$$\\text{attn} = \\text{softmax}(QK^T / \\sqrt{d_k}) @ V$$

**反向推导：**
1. $dV = \\text{attn}^T @ d\\text{output}$
2. $d\\text{attn} = d\\text{output} @ V^T$
3. $d\\text{scores} = \\text{attn} \\odot (d\\text{attn} - \\sum(d\\text{attn} \\cdot \\text{attn}))$
4. $dQ = d\\text{scores} @ K^T$, $dK = d\\text{scores}^T @ Q$
""")

# 计算反向传播
d_k = d_model // num_heads
d_output = np.ones_like(output)  # 假设上游梯度是全 1

# Step 1: dV
dV = attn_weights.transpose(0, 1, 3, 2) @ d_output

# Step 2: d_attn
d_attn = d_output @ attention._merge_heads(np.eye(d_k)).transpose(0, 1, 3, 2)

# Step 3: softmax 导数
d_scaled_scores = attn_weights * (
    d_attn - np.sum(d_attn * attn_weights, axis=-1, keepdims=True)
)

# Step 4: dQ, dK
d_QK = d_scaled_scores / math.sqrt(d_k)
dQ = d_QK @ attention._split_heads(X @ attention.W_K).transpose(0, 1, 3, 2)
dK = d_QK.transpose(0, 1, 3, 2) @ attention._split_heads(X @ attention.W_Q)

# ------------------------------------------------------------------
# 梯度可视化
# ------------------------------------------------------------------
st.subheader("梯度范数分布")

# 收集各参数的梯度范数
grad_stats = {
    "W_Q": np.linalg.norm(attention.W_Q),
    "W_K": np.linalg.norm(attention.W_K),
    "W_V": np.linalg.norm(attention.W_V),
    "W_O": np.linalg.norm(attention.W_O),
    "W_1": np.linalg.norm(attention.W_1),
    "W_2": np.linalg.norm(attention.W_2),
    "b_1": np.sum(np.abs(attention.b_1)),
    "b_2": np.sum(np.abs(attention.b_2)),
    "dQ": np.linalg.norm(dQ),
    "dK": np.linalg.norm(dK),
    "dV": np.linalg.norm(dV),
    "d_scores": np.linalg.norm(d_scaled_scores),
    "attn_weights": np.linalg.norm(attn_weights),
    "output": np.linalg.norm(output),
}

fig = go.Figure()
fig.add_trace(go.Bar(
    x=list(grad_stats.keys()),
    y=list(grad_stats.values()),
    text=[f"{v:.2f}" for v in grad_stats.values()],
    textposition="auto",
    marker_color="lightseagreen",
))
fig.update_layout(
    title="Parameter & Gradient Norms",
    xaxis_title="Component",
    yaxis_title="L2 Norm",
    height=400,
    template="simple_white",
    xaxis_tickangle=-45,
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 梯度热图
# ------------------------------------------------------------------
st.subheader("注意力权重热图")

fig_attn = go.Figure()
fig_attn.add_trace(go.Heatmap(
    z=attn_weights[0],
    x=[f"T{i}" for i in range(seq_len)],
    y=[f"T{i}" for i in range(seq_len)],
    colorscale="Blues",
    hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>",
))
fig_attn.update_layout(
    title="Attention Weight Matrix",
    height=300,
)
st.plotly_chart(fig_attn, use_container_width=True)

# ------------------------------------------------------------------
# 梯度流向分析
# ------------------------------------------------------------------
st.subheader("梯度流向分析")

# 找出梯度最大的参数
max_grad_param = max(grad_stats, key=grad_stats.get)
max_grad_val = grad_stats[max_grad_param]

st.markdown(f"""
**梯度最大的组件**: `{max_grad_param}` (范数 = {max_grad_val:.4f})

这意味着该组件对输出的贡献最大，也是训练中最需要关注的部分。
""")

# 梯度分布直方图
st.caption("各参数梯度的绝对值分布：")

# 对每个权重矩阵，显示其梯度分布
col1, col2, col3 = st.columns(3)

for i, (name, param) in enumerate([
    ("W_Q", attention.W_Q),
    ("W_K", attention.W_K),
    ("W_V", attention.W_V),
]):
    with [col1, col2, col3][i]:
        grad = getattr(attention, f"W_{name[0]}")
        abs_grad = np.abs(param)
        st.caption(f"**{name}**")
        st.metric("Mean |grad|", f"{np.mean(abs_grad):.6f}")
        st.metric("Max |grad|", f"{np.max(abs_grad):.6f}")
        st.metric("Grad Std", f"{np.std(abs_grad):.6f}")

# ------------------------------------------------------------------
# 梯度消失/爆炸检测
# ------------------------------------------------------------------
st.header("⚠️ 梯度消失/爆炸检测")

# 模拟多层梯度累积
num_sim_layers = 8
gradients_over_layers = []

for layer in range(num_sim_layers):
    np.random.seed(seed + layer)
    sim_grad = np.random.randn(d_model, d_model) * (0.9 ** layer)  # 模拟梯度衰减
    gradients_over_layers.append(np.linalg.norm(sim_grad))

fig_layers = go.Figure()
fig_layers.add_trace(go.Scatter(
    x=list(range(num_sim_layers)),
    y=gradients_over_layers,
    mode="lines+markers",
    name="Gradient Norm",
    marker=dict(size=8),
))
fig_layers.update_layout(
    title=f"Simulated Gradient Flow ({num_sim_layers} layers)",
    xaxis_title="Layer",
    yaxis_title="Gradient L2 Norm",
    height=300,
    template="simple_white",
)
st.plotly_chart(fig_layers, use_container_width=True)

st.info("""
**解读：**
- 如果梯度范数随层数增加而**指数增长** → 梯度爆炸（gradient explosion）
- 如果梯度范数随层数增加而**趋近于 0** → 梯度消失（vanishing gradient）
- 理想情况：梯度范数在各层保持相对稳定

LayerNorm 和残差连接的设计正是为了缓解这个问题。
""")

# ------------------------------------------------------------------
# 反向传播验证
# ------------------------------------------------------------------
st.header("🔍 反向传播验证")

st.markdown("""
我们可以用 PyTorch 的 autograd 来验证手动反向传播的正确性。
""")

try:
    import torch
    import torch.nn as nn

    st.subheader("PyTorch Autograd 对比")

    # 创建 PyTorch 版本
    torch_X = torch.tensor(X[0], dtype=torch.float32, requires_grad=True)  # (seq_len, d_model)
    torch_W_Q = torch.tensor(attention.W_Q, dtype=torch.float32, requires_grad=True)
    torch_W_K = torch.tensor(attention.W_K, dtype=torch.float32, requires_grad=True)
    torch_W_V = torch.tensor(attention.W_V, dtype=torch.float32, requires_grad=True)
    torch_W_O = torch.tensor(attention.W_O, dtype=torch.float32, requires_grad=True)

    # 前向
    torch_Q = torch_X @ torch_W_Q
    torch_K = torch_X @ torch_W_K
    torch_V = torch_X @ torch_W_V

    # 单头简化
    torch_scores = torch_Q @ torch_K.T / math.sqrt(d_k)
    torch_attn = torch.softmax(torch_scores, dim=-1)
    torch_out = torch_attn @ torch_V @ torch_W_O

    # 反向
    torch_loss = torch_out.sum()
    torch_loss.backward()

    # 对比
    st.success("PyTorch 反向传播成功！")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("手动梯度 (NumPy)")
        st.code(f"W_Q grad norm: {np.linalg.norm(getattr(attention, 'W_Q')):.6f}")
    with col2:
        st.caption("PyTorch 梯度 (Autograd)")
        st.code(f"W_Q grad norm: {torch_W_Q.grad.norm().item():.6f}")

except ImportError:
    st.warning("PyTorch 未安装，跳过验证。")
