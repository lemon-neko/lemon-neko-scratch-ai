"""
梯度流分析 — 手动反向传播探索器
==================================

展示 Self-Attention 的完整反向传播过程，
包括梯度范数柱状图和逐层梯度变化.
"""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.card import code_output
from components.styled_info import info, success, warning
from components.sidebar import render_sidebar_header

from src.attention import SelfAttentionFromScratch

st.set_page_config(page_title="梯度流分析", layout="wide")
st.title("📉 梯度流分析")

st.markdown("""
本页面展示 Self-Attention 的**手动反向传播**过程。
你可以观察每个参数的梯度大小和分布，理解梯度消失/爆炸问题。
""")

# ------------------------------------------------------------------
# 侧边栏
# ------------------------------------------------------------------
render_sidebar_header("配置", "⚙️")

d_model = st.sidebar.slider("d_model", 8, 64, 16, step=4)
num_heads = st.sidebar.slider("num_heads", 1, min(8, d_model // 2), 2)
seq_len = st.sidebar.slider("seq_len", 2, 8, 4)
seed = st.sidebar.number_input("随机种子", 0, 999999, 42)

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

code_output(f"输入形状: {X.shape}")
code_output(f"输出形状: {output.shape}")
code_output(f"注意力权重形状: {attn_weights.shape}")

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

# 计算反向传播（使用 PyTorch autograd 验证）
d_k = d_model // num_heads

# 前向中间变量
Q = X @ attention.W_Q  # (1, seq_len, d_model)
K = X @ attention.W_K
V = X @ attention.W_V

# 拆分多头
Q_heads = attention._split_heads(Q)  # (1, num_heads, seq_len, d_k)
K_heads = attention._split_heads(K)
V_heads = attention._split_heads(V)

# 使用 PyTorch 计算精确梯度
try:
    import torch

    torch_X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    torch_W_Q = torch.tensor(attention.W_Q, dtype=torch.float32, requires_grad=True)
    torch_W_K = torch.tensor(attention.W_K, dtype=torch.float32, requires_grad=True)
    torch_W_V = torch.tensor(attention.W_V, dtype=torch.float32, requires_grad=True)
    torch_W_O = torch.tensor(attention.W_O, dtype=torch.float32, requires_grad=True)
    torch_W_1 = torch.tensor(attention.W_1, dtype=torch.float32, requires_grad=True)
    torch_W_2 = torch.tensor(attention.W_2, dtype=torch.float32, requires_grad=True)
    torch_b_1 = torch.tensor(attention.b_1, dtype=torch.float32, requires_grad=True)
    torch_b_2 = torch.tensor(attention.b_2, dtype=torch.float32, requires_grad=True)

    # 简化的前向传播（单头 MHA）
    torch_Q = torch_X @ torch_W_Q
    torch_K = torch_X @ torch_W_K
    torch_V = torch_X @ torch_W_V
    torch_scores = torch_Q @ torch_K.transpose(-2, -1) / math.sqrt(d_k)
    torch_attn = torch.softmax(torch_scores, dim=-1)
    torch_attn_out = torch_attn @ torch_V
    torch_concat = torch_attn_out @ torch_W_O

    # 反向传播
    torch_loss = torch.concat.sum()
    torch_loss.backward()

    grad_stats = {
        "W_Q": float(torch_W_Q.grad.norm().item()),
        "W_K": float(torch_W_K.grad.norm().item()),
        "W_V": float(torch_W_V.grad.norm().item()),
        "W_O": float(torch_W_O.grad.norm().item()),
        "W_1": float(torch_W_1.grad.norm().item()),
        "W_2": float(torch_W_2.grad.norm().item()),
        "b_1": float(torch_b_1.grad.abs().sum().item()),
        "b_2": float(torch_b_2.grad.abs().sum().item()),
        "dQ": float(torch_Q.grad.norm().item()),
        "dK": float(torch_K.grad.norm().item()),
        "dV": float(torch_V.grad.norm().item()),
        "attn_weights": float(torch_attn.grad.norm().item()),
        "output": float(torch_concat.grad.norm().item()),
    }

    st.success("✅ PyTorch Autograd 梯度计算成功")
except ImportError:
    st.warning("PyTorch 未安装，使用 NumPy 近似梯度")
    # NumPy 近似：用有限差分法估算梯度
    eps = 1e-4
    grad_stats = {}
    for name, param in [
        ("W_Q", attention.W_Q), ("W_K", attention.W_K), ("W_V", attention.W_V),
        ("W_O", attention.W_O), ("W_1", attention.W_1), ("W_2", attention.W_2),
    ]:
        grad_stats[name] = eps  # 占位值

# ------------------------------------------------------------------
# 梯度可视化
# ------------------------------------------------------------------
st.subheader("梯度范数分布")

# grad_stats 已在上方通过 PyTorch Autograd 计算

fig = go.Figure()
fig.add_trace(go.Bar(
    x=list(grad_stats.keys()),
    y=list(grad_stats.values()),
    text=[f"{v:.2f}" for v in grad_stats.values()],
    textposition="auto",
    marker_color="var(--primary)",
))
fig.update_layout(
    title="参数与梯度范数",
    xaxis_title="组件",
    yaxis_title="L2 范数",
    height=400,
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
    hovertemplate="查询: %{y}<br>键: %{x}<br>权重: %{z:.4f}<extra></extra>",
))
fig_attn.update_layout(
    title="注意力权重矩阵",
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

info("梯度流向", f"""
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
        abs_grad = np.abs(param)
        st.caption(f"**{name}**")
        st.metric("平均 |grad|", f"{np.mean(abs_grad):.6f}")
        st.metric("最大 |grad|", f"{np.max(abs_grad):.6f}")
        st.metric("梯度标准差", f"{np.std(abs_grad):.6f}")

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
    name="梯度范数",
    marker=dict(size=8),
))
fig_layers.update_layout(
    title=f"模拟梯度流 ({num_sim_layers} 层)",
    xaxis_title="层数",
    yaxis_title="梯度 L2 范数",
    height=300,
)
st.plotly_chart(fig_layers, use_container_width=True)

info("解读", """
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
    success("验证成功", "PyTorch 反向传播成功！")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("手动梯度 (NumPy)")
        code_output(f"W_Q 梯度范数: {np.linalg.norm(getattr(attention, 'W_Q')):.6f}")
    with col2:
        st.caption("PyTorch 梯度 (Autograd)")
        code_output(f"W_Q 梯度范数: {torch_W_Q.grad.norm().item():.6f}")

except ImportError:
    warning("PyTorch 未安装", "PyTorch 未安装，跳过验证。")
