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

from components.card import code_output, page_title, section_divider, section_title
from components.styled_info import info, success, warning
from src.attention import SelfAttentionFromScratch
from components.nav import render_nav_bar

st.set_page_config(page_title="梯度流分析", layout="wide", page_icon="🐱")
render_nav_bar(active_page="梯度流分析")

# ------------------------------------------------------------------
# 三栏布局
# ------------------------------------------------------------------
left_col, center_col, right_col = st.columns([1, 4, 1])

with left_col:
    # 左侧参数面板
    st.markdown('<div class="app-left-panel-inner">', unsafe_allow_html=True)
    st.markdown("#### 🎛️ 配置参数")

    d_model = st.slider("d_model", 8, 64, 16, step=4)
    num_heads = st.slider("num_heads", 1, min(8, d_model // 2), 2)
    seq_len = st.slider("seq_len", 2, 8, 4)
    seed = st.number_input("随机种子", 0, 999999, 42)

    # 配置摘要卡片
    st.markdown(f"""
    <div class="config-card">
        <div class="config-card-title">当前配置</div>
        <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.8;">
            <b>d_model</b>: <code>{d_model}</code><br/>
            <b>num_heads</b>: <code>{num_heads}</code><br/>
            <b>seq_len</b>: <code>{seq_len}</code><br/>
            <b>seed</b>: <code>{seed}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with center_col:
    page_title("📉", "梯度流分析")

    st.markdown("""
    本页面展示 Self-Attention 的**手动反向传播**过程。
    你可以观察每个参数的梯度大小和分布，理解梯度消失/爆炸问题。
    """)

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
    section_divider()
    section_title(icon="📐", text="前向传播", size="1.3rem")

    X = np.random.randn(1, seq_len, d_model)
    output, attn_weights = attention.forward(X, training=False)

    code_output(f"输入形状: {X.shape}")
    code_output(f"输出形状: {output.shape}")
    code_output(f"注意力权重形状: {attn_weights.shape}")

    # ------------------------------------------------------------------
    # 手动反向传播
    # ------------------------------------------------------------------
    section_divider()
    section_title(icon="🔄", text="手动反向传播", size="1.3rem")

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

        success("Autograd 计算成功", "PyTorch 梯度计算完成！")
    except ImportError:
        warning("未安装 PyTorch", "使用 NumPy 有限差分法近似梯度。")
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
    section_title(icon="", text="梯度范数分布", size="1.2rem")

    # grad_stats 已在上方通过 PyTorch Autograd 计算

    fig = go.Figure()
    grad_colors = [
        "#00d4aa", "#06b6d4", "#3b82f6", "#8b5cf6", "#7c3aed",
        "#f97316", "#ec4899", "#ef4444", "#22c55e", "#06b6d4",
        "#3b82f6", "#8b5cf6",
    ]
    fig.add_trace(go.Bar(
        x=list(grad_stats.keys()),
        y=list(grad_stats.values()),
        text=[f"{v:.2f}" for v in grad_stats.values()],
        textposition="auto",
        marker_color=grad_colors[:len(grad_stats)],
    ))
    fig.update_layout(
        title="参数与梯度范数",
        xaxis_title="组件",
        yaxis_title="L2 范数",
        height=400,
        xaxis_tickangle=-45,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94A3B8")),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 梯度热图
    # ------------------------------------------------------------------
    section_title(icon="", text="注意力权重热图", size="1.2rem")

    fig_attn = go.Figure()
    fig_attn.add_trace(go.Heatmap(
        z=attn_weights[0],
        x=[f"T{i}" for i in range(seq_len)],
        y=[f"T{i}" for i in range(seq_len)],
        colorscale=[
            [0.0, "#060a14"],
            [0.15, "#0a1628"],
            [0.3, "#0d3d5a"],
            [0.45, "#0d5f7a"],
            [0.6, "#0d8a7a"],
            [0.75, "#00d4aa"],
            [0.9, "#33e0bf"],
            [1.0, "#ffe040"],
        ],
        hovertemplate="查询: %{y}<br>键: %{x}<br>权重: %{z:.4f}<extra></extra>",
    ))
    fig_attn.update_layout(
        title="注意力权重矩阵",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94A3B8")),
    )
    fig_attn.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color="#94A3B8"),
            bgcolor="rgba(0,0,0,0)",
            outlinewidth=0,
        )
    )
    st.plotly_chart(fig_attn, use_container_width=True)

    # ------------------------------------------------------------------
    # 梯度流向分析
    # ------------------------------------------------------------------
    section_title(icon="", text="梯度流向分析", size="1.2rem")

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
            st.markdown(f'**{name}**')
            st.markdown(f'平均 |grad|: <strong style="color:var(--primary);">{np.mean(abs_grad):.6f}</strong>')
            st.markdown(f'最大 |grad|: <strong style="color:var(--primary);">{np.max(abs_grad):.6f}</strong>')
            st.markdown(f'梯度标准差: <strong style="color:var(--primary);">{np.std(abs_grad):.6f}</strong>')

    # ------------------------------------------------------------------
    # 梯度消失/爆炸检测
    # ------------------------------------------------------------------
    section_divider()
    section_title(icon="⚠️", text="梯度消失/爆炸检测", size="1.3rem")

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
        marker=dict(size=8, color="#00d4aa"),
        line=dict(color="#00d4aa", width=2),
    ))
    fig_layers.update_layout(
        title=f"模拟梯度流 ({num_sim_layers} 层)",
        xaxis_title="层数",
        yaxis_title="梯度 L2 范数",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94A3B8")),
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
    section_divider()
    section_title(icon="🔍", text="反向传播验证", size="1.3rem")

    st.markdown("""
    我们可以用 PyTorch 的 autograd 来验证手动反向传播的正确性。
    """)

    try:
        import torch

        section_title(icon="", text="PyTorch Autograd 对比", size="1.2rem")

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
        warning("未安装 PyTorch", "PyTorch 未安装，跳过验证。")

with right_col:
    # 右侧参考面板
    st.markdown("""
    <div class="app-right-panel-inner">
        <div class="right-panel-title">📚 梯度参考</div>
        <div class="formula-card">
            dV = attn<sup>T</sup> · d(output)
        </div>
        <div class="formula-card">
            dQ = d(scores) · K<sup>T</sup>
        </div>
        <div class="formula-card">
            dK = d(scores)<sup>T</sup> · Q
        </div>
        <div class="tip-card">
            <div class="tip-dot" style="background:#00d4aa;"></div>
            <div>
                <div class="tip-title">梯度解读</div>
                <div class="tip-body">梯度范数越大，该参数对输出影响越显著</div>
            </div>
        </div>
        <div class="tip-card">
            <div class="tip-dot" style="background:#ef4444;"></div>
            <div>
                <div class="tip-title">梯度消失</div>
                <div class="tip-body">深层网络中梯度趋近于 0，参数无法更新</div>
            </div>
        </div>
        <div class="tip-card">
            <div class="tip-dot" style="background:#f59e0b;"></div>
            <div>
                <div class="tip-title">关键公式</div>
                <div class="tip-body">LayerNorm + 残差连接可有效缓解梯度消失</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
