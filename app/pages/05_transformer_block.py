"""
Transformer 块可视化
=====================

展示完整 Encoder/Decoder Block 的结构，包括每一步的张量形状标注。
"""

import numpy as np

import streamlit as st

st.set_page_config(page_title="Transformer 块", layout="wide")
st.title("🧱 Transformer Block 可视化")

st.markdown("""
本页面展示 Transformer Encoder 和 Decoder Block 的完整结构，
每一步都标注了张量形状的变化。
""")

# ------------------------------------------------------------------
# 侧边栏：模型配置
# ------------------------------------------------------------------
st.sidebar.header("⚙️ 模型配置")

d_model = st.sidebar.slider("d_model", 32, 256, 64, step=8)
num_heads = st.sidebar.slider("num_heads", 1, min(8, d_model // 4), 4)
d_ff = st.sidebar.slider("d_ff", 32, 512, d_model * 2, step=8)
num_layers = st.sidebar.slider("num_layers", 1, 6, 2)

d_k = d_model // num_heads
d_v = d_k

st.sidebar.markdown(f"**d_k = {d_k}, d_v = {d_v}**")

# ------------------------------------------------------------------
# Encoder Block 结构
# ------------------------------------------------------------------
st.header("📐 Encoder Block")

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

# ---- 逐步展示 ----
st.subheader("逐步张量形状")

steps = [
    ("输入 X", f"({num_layers}, {d_model // num_layers}, {d_model})" if num_layers > 1 else f"(1, {d_model}, {d_model})"),
    ("Q = XW_Q", f"({num_layers}, {d_model // num_layers}, {d_model})"),
    ("K = XW_K", f"({num_layers}, {d_model // num_layers}, {d_model})"),
    ("V = XW_V", f"({num_layers}, {d_model // num_layers}, {d_model})"),
    ("拆分多头 Q", f"({num_layers}, {num_heads}, {d_model // num_layers}, {d_k})"),
    ("拆分多头 K", f"({num_layers}, {num_heads}, {d_model // num_layers}, {d_k})"),
    ("拆分多头 V", f"({num_layers}, {num_heads}, {d_model // num_layers}, {d_k})"),
    ("分数 = QK^T / √d_k", f"({num_layers}, {num_heads}, {d_model // num_layers}, {d_model // num_layers})"),
    ("注意力权重", f"({num_layers}, {num_heads}, {d_model // num_layers}, {d_model // num_layers})"),
    ("注意力 @ V", f"({num_layers}, {num_heads}, {d_model // num_layers}, {d_k})"),
    ("合并多头", f"({num_layers}, {d_model // num_layers}, {d_model})"),
    ("输出 @ W_O", f"({num_layers}, {d_model // num_layers}, {d_model})"),
    ("残差 + X", f"({num_layers}, {d_model // num_layers}, {d_model})"),
    ("LayerNorm", f"({num_layers}, {d_model // num_layers}, {d_model})"),
    ("FFN: 线性(d_model → d_ff)", f"({num_layers}, {d_model // num_layers}, {d_ff})"),
    ("FFN: ReLU", f"({num_layers}, {d_model // num_layers}, {d_ff})"),
    ("FFN: 线性(d_ff → d_model)", f"({num_layers}, {d_model // num_layers}, {d_model})"),
    ("最终 LayerNorm", f"({num_layers}, {d_model // num_layers}, {d_model})"),
]

for name, shape in steps:
    st.caption(f"**{name}**: `{shape}`")

# ---- 参数量计算 ----
st.subheader("参数量统计")

total_params = (
    d_model * d_model * 4  # W_Q, W_K, W_V, W_O
    + d_model * d_ff * 2   # W_1, W_2
    + d_ff + d_model       # b_1, b_2
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Attention 权重", f"{d_model * d_model * 4:,}")
with col2:
    st.metric("FFN 权重", f"{d_model * d_ff * 2 + d_ff + d_model:,}")
with col3:
    st.metric("单 Block 总计", f"{total_params:,}")
with col1:
    st.metric(f"总参数量 ({num_layers} 层)", f"{total_params * num_layers:,}")

# ------------------------------------------------------------------
# Decoder Block 结构
# ------------------------------------------------------------------
st.header("📐 Decoder Block")

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

st.info("""
**Encoder-Decoder 区别：**
- **Encoder**: Self-Attention → FFN
- **Decoder**: Masked Self-Attention → Cross-Attention → FFN
- **Masked Self-Attention**: 使用 causal mask 防止看到未来 token
- **Cross-Attention**: Q 来自 decoder, K/V 来自 encoder 输出
""")

# ------------------------------------------------------------------
# 交互式对比
# ------------------------------------------------------------------
st.header("🔄 Encoder vs Decoder 对比")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Encoder Block")
    st.markdown("""
    - **Self-Attention**: Q, K, V 都来自同一输入
    - **FFN**: 逐位置独立变换
    - **Mask**: 无（可以看到所有 token）
    - **用途**: 理解输入序列的内部关系
    """)

with col2:
    st.subheader("Decoder Block")
    st.markdown("""
    - **Masked Self-Attention**: 因果 mask 保护自回归
    - **Cross-Attention**: Q 来自 decoder, K/V 来自 encoder
    - **FFN**: 逐位置独立变换
    - **用途**: 结合输入信息和已生成内容
    """)

# ------------------------------------------------------------------
# 位置编码
# ------------------------------------------------------------------
st.header("📍 位置编码")

st.markdown("""
由于 Self-Attention 是排列不变的，必须通过位置编码注入位置信息。

$$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$
$$PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$
""")

pe = np.zeros((10, d_model))
pos = np.arange(10)[:, np.newaxis]
div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000) / d_model))
pe[:, 0::2] = np.sin(pos * div_term)
pe[:, 1::2] = np.cos(pos * div_term)

st.dataframe(pe.round(4), use_container_width=True)

st.caption("上图展示了前 10 个位置的编码向量（d_model 维）。相邻位置的编码向量相似度高，远距离的位置编码差异大。")
