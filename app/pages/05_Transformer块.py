"""
Transformer 块可视化
=====================

展示完整 Encoder/Decoder Block 的结构，包括每一步的张量形状标注.
"""

import numpy as np

import streamlit as st
from components.card import comparison_card, info_panel, page_title, section_divider, metric_row, section_title
from components.nav import render_nav_bar

st.set_page_config(page_title="Transformer 块", layout="wide", page_icon="🐱")
render_nav_bar(active_page="Transformer 块")

# ------------------------------------------------------------------
# 三栏布局
# ------------------------------------------------------------------
left_col, center_col, right_col = st.columns([1, 4, 1])

with left_col:
    # 左侧参数面板
    st.markdown('<div class="app-left-panel-inner">', unsafe_allow_html=True)
    st.markdown("#### 🎛️ 模型配置")

    d_model = st.slider("d_model", 32, 256, 64, step=8)
    num_heads = st.slider("num_heads", 1, min(8, d_model // 4), 4)
    d_ff = st.slider("d_ff", 32, 512, d_model * 2, step=8)
    num_layers = st.slider("num_layers", 1, 6, 2)
    seq_len = 4  # 示例序列长度

    d_k = d_model // num_heads
    d_v = d_k

    # 配置摘要卡片（包含 d_k/d_v 计算指标）
    st.markdown(f"""
    <div class="config-card">
        <div class="config-card-title">当前配置</div>
        <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.8;">
            <b>d_model</b>: <code>{d_model}</code><br/>
            <b>num_heads</b>: <code>{num_heads}</code><br/>
            <b>d_k</b>: <code>{d_k}</code><br/>
            <b>d_v</b>: <code>{d_v}</code><br/>
            <b>d_ff</b>: <code>{d_ff}</code><br/>
            <b>num_layers</b>: <code>{num_layers}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with center_col:
    page_title("🧱", "Transformer 块")

    st.markdown("""
    本页面展示 Transformer Encoder 和 Decoder Block 的完整结构，
    每一步都标注了张量形状的变化。
    """)

    # ------------------------------------------------------------------
    # Encoder Block 结构
    # ------------------------------------------------------------------
    section_title(icon="📐", text="Encoder Block", size="1.5rem")

    st.markdown("""
    Encoder Block 由两个子层组成：
    1. **Multi-Head Self-Attention**
    2. **Position-wise Feed Forward Network**

    每个子层都有残差连接和 Layer Normalization。

    ```
    Input (batch, seq_len, d_model)
        │
        ├─► Multi-Head Attention ──► (batch, seq_len, d_model)
        │        │                       │
        │        │              Residual (+)
        │        │                       │
        │        └───── X ───────────────┘
        │                               │
        │                        LayerNorm
        │                               │
        │                    Hidden (batch, seq_len, d_model)
        │                               │
        │                       FFN: d_model → d_ff → d_model
        │                               │
        │                       Residual (+)
        │                               │
        │                        LayerNorm
        │                               │
        ▼                      Output (batch, seq_len, d_model)
    ```
    """)

    # ---- 逐步展示（styled HTML table） ----
    section_title(icon="", text="逐步张量形状", size="1.2rem")

    steps = [
        ("输入 X", f"({1}, {seq_len}, {d_model})"),
        ("Q = XW_Q", f"({1}, {seq_len}, {d_model})"),
        ("K = XW_K", f"({1}, {seq_len}, {d_model})"),
        ("V = XW_V", f"({1}, {seq_len}, {d_model})"),
        ("拆分多头 Q", f"({1}, {num_heads}, {seq_len}, {d_k})"),
        ("拆分多头 K", f"({1}, {num_heads}, {seq_len}, {d_k})"),
        ("拆分多头 V", f"({1}, {num_heads}, {seq_len}, {d_k})"),
        ("分数 = QK^T / √d_k", f"({1}, {num_heads}, {seq_len}, {seq_len})"),
        ("注意力权重", f"({1}, {num_heads}, {seq_len}, {seq_len})"),
        ("注意力 @ V", f"({1}, {num_heads}, {seq_len}, {d_k})"),
        ("合并多头", f"({1}, {seq_len}, {d_model})"),
        ("输出 @ W_O", f"({1}, {seq_len}, {d_model})"),
        ("残差 + X", f"({1}, {seq_len}, {d_model})"),
        ("LayerNorm", f"({1}, {seq_len}, {d_model})"),
        ("FFN: 线性(d_model → d_ff)", f"({1}, {seq_len}, {d_ff})"),
        ("FFN: ReLU", f"({1}, {seq_len}, {d_ff})"),
        ("FFN: 线性(d_ff → d_model)", f"({1}, {seq_len}, {d_model})"),
        ("最终 LayerNorm", f"({1}, {seq_len}, {d_model})"),
    ]

    table_rows = "".join(
        f'<tr><td style="padding:0.5rem 1rem;border-bottom:1px solid var(--border);color:var(--text-secondary);">{i + 1}</td>'
        f'<td style="padding:0.5rem 1rem;border-bottom:1px solid var(--border);font-weight:600;color:var(--text-primary);">{name}</td>'
        f'<td style="padding:0.5rem 1rem;border-bottom:1px solid var(--border);font-family:monospace;color:var(--primary);">{shape}</td></tr>'
        for i, (name, shape) in enumerate(steps)
    )

    st.markdown(f"""
    <div style="background:var(--surface);border-radius:10px;border:1px solid var(--border);overflow:hidden;margin:0.5rem 0;">
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
    <thead><tr style="background:var(--surface-alt);border-bottom:2px solid var(--border);">
    <th style="padding:0.75rem 1rem;text-align:left;width:3rem;">#</th>
    <th style="padding:0.75rem 1rem;text-align:left;">步骤</th>
    <th style="padding:0.75rem 1rem;text-align:left;">张量形状</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
    </table></div>
    """, unsafe_allow_html=True)

    # ---- 参数量计算（styled metric cards） ----
    section_title(icon="", text="参数量统计", size="1.2rem")

    total_params = (
        d_model * d_model * 4  # W_Q, W_K, W_V, W_O
        + d_model * d_ff * 2   # W_1, W_2
        + d_ff + d_model       # b_1, b_2
    )

    metric_row([
        {"label": "Attention 权重", "value": f"{d_model * d_model * 4:,}"},
        {"label": "FFN 权重", "value": f"{d_model * d_ff * 2 + d_ff + d_model:,}"},
        {"label": "单 Block 总计", "value": f"{total_params:,}"},
        {"label": f"总参数量 ({num_layers} 层)", "value": f"{total_params * num_layers:,}", "color": "var(--success)"},
    ])

    section_divider()
    section_title(icon="📐", text="Decoder Block", size="1.5rem")

    st.markdown("""
    Decoder Block 比 Encoder 多了一个 **Cross-Attention** 子层：

    ```
    Input (batch, tgt_len, d_model)
        │
        ├─► Masked Multi-Head Attn ──► (batch, tgt_len, d_model)
        │        │
        │        ├─ Residual + LN
        │        │
        │        ├─► Multi-Head Attn (encoder output as K, V) ──► (batch, tgt_len, d_model)
        │        │        │
        │        │        ├─ Residual + LN
        │        │        │
        │        │        ├─► FFN ──► (batch, tgt_len, d_model)
        │        │                │
        │        │                ├─ Residual + LN
        │        │                │
        │        ▼                ▼
        ▼                      Output (batch, tgt_len, d_model)
    ```
    """)

    info_panel(
        content="""
        <strong>Encoder-Decoder 区别：</strong>
        <ul style="margin:0.5rem 0 0 0;padding-left:1.25rem;">
            <li><strong>Encoder</strong>: Self-Attention → FFN</li>
            <li><strong>Decoder</strong>: Masked Self-Attention → Cross-Attention → FFN</li>
            <li><strong>Masked Self-Attention</strong>: 使用 causal mask 防止看到未来 token</li>
            <li><strong>Cross-Attention</strong>: Q 来自 decoder, K/V 来自 encoder 输出</li>
        </ul>
        """,
    )

    section_divider()
    section_title(icon="🔄", text="Encoder vs Decoder 对比", size="1.5rem")

    col1, col2 = st.columns(2)

    with col1:
        comparison_card(
            "Encoder Block",
            "📐",
            [
                "<strong>Self-Attention</strong>: Q, K, V 都来自同一输入",
                "<strong>FFN</strong>: 逐位置独立变换",
                "<strong>Mask</strong>: 无（可以看到所有 token）",
                "<strong>用途</strong>: 理解输入序列的内部关系",
            ],
        )

    with col2:
        comparison_card(
            "Decoder Block",
            "📐",
            [
                "<strong>Masked Self-Attention</strong>: 因果 mask 保护自回归",
                "<strong>Cross-Attention</strong>: Q 来自 decoder, K/V 来自 encoder",
                "<strong>FFN</strong>: 逐位置独立变换",
                "<strong>用途</strong>: 结合输入信息和已生成内容",
            ],
        )

    section_divider()
    section_title(icon="📍", text="位置编码", size="1.5rem")

    info_panel(
        content="""
        <strong>正弦位置编码</strong>
        <p style="color:var(--text-secondary);font-size:0.95rem;line-height:1.7;margin-bottom:1rem;">
        由于 Self-Attention 是排列不变的，必须通过位置编码注入位置信息。
        </p>
        $$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$
        $$PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$
        """,
    )

    pe = np.zeros((10, d_model))
    pos = np.arange(10)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000) / d_model))
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)

    st.dataframe(pe.round(4), use_container_width=True)

    st.caption("上图展示了前 10 个位置的编码向量（d_model 维）。相邻位置的编码向量相似度高，远距离的位置编码差异大。")

with right_col:
    # 右侧参考面板
    st.markdown("""
    <div class="app-right-panel-inner">
        <div class="right-panel-title">📚 公式速查</div>
        <div class="formula-card">
            MHA = Concat(head<sub>i</sub>) W<sup>O</sup>
        </div>
        <div class="formula-card">
            FFN(x) = ReLU(xW<sub>1</sub>+b<sub>1</sub>)W<sub>2</sub>+b<sub>2</sub>
        </div>
        <div class="formula-card">
            LN(x) = γ · (x-μ)/σ + β
        </div>
        <div class="tip-card">
            <div class="tip-dot" style="background:#00d4aa;"></div>
            <div>
                <div class="tip-title">残差连接</div>
                <div class="tip-body">解决深层网络梯度消失问题</div>
            </div>
        </div>
        <div class="tip-card">
            <div class="tip-dot" style="background:#3b82f6;"></div>
            <div>
                <div class="tip-title">LayerNorm</div>
                <div class="tip-body">对每个样本的特征维度做归一化</div>
            </div>
        </div>
        <div class="tip-card">
            <div class="tip-dot" style="background:#f59e0b;"></div>
            <div>
                <div class="tip-title">Cross-Attention</div>
                <div class="tip-body">Q 来自 decoder，K/V 来自 encoder</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
