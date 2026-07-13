"""
注意力探索器 — 交互式注意力权重可视化
"""

import math

import numpy as np
import streamlit as st

from components.attention_heatmap import render_attention_heatmap
from components.card import badge, code_output, page_title, section_divider, section_title
from components.nav import render_nav_bar

st.set_page_config(page_title="注意力探索器", layout="wide", page_icon="🐱")
render_nav_bar(active_page="注意力热力图")

# ------------------------------------------------------------------
# 三栏布局
# ------------------------------------------------------------------
left_col, center_col, right_col = st.columns([1, 4, 1])

with left_col:
    # 左侧参数面板
    st.markdown('<div class="app-left-panel-inner">', unsafe_allow_html=True)
    st.markdown("#### 🎛️ 配置参数")

    input_text = st.text_area("输入文本", "我爱看猫", height=60)
    tokens = list(input_text.strip()) if input_text else ["我", "爱", "看", "猫"]
    d_model = st.slider("d_model", 4, 128, 16, step=4)
    num_heads = st.slider("num_heads", 1, min(8, d_model // 2), 4)
    temperature = st.slider("温度系数", 0.1, 5.0, 1.0, 0.1)
    init_method = st.selectbox("权重初始化方法", ["Xavier", "Kaiming", "Random"])

    d_k = d_model // num_heads

    # 配置摘要卡片
    st.markdown(f"""
    <div class="config-card">
        <div class="config-card-title">当前配置</div>
        <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.8;">
            <b>tokens</b>: <code>{len(tokens)}</code><br/>
            <b>d_k</b>: <code>{d_k}</code><br/>
            <b>温度</b>: <code>{temperature}</code><br/>
            <b>初始化</b>: <code>{init_method}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with center_col:
    page_title("🔍", "注意力探索器")

    st.markdown("""
    输入文本，探索不同配置下的注意力权重分布。
    你可以调节 temperature 来改变注意力的尖锐程度。
    """)

    # ------------------------------------------------------------------
    # 生成随机权重
    # ------------------------------------------------------------------
    np.random.seed(42)

    if init_method == "Xavier":
        scale = 1.0 / math.sqrt(d_model)
    elif init_method == "Kaiming":
        scale = math.sqrt(2.0 / d_model)
    else:
        scale = 0.1

    W_Q = np.random.randn(d_model, d_model) * scale
    W_K = np.random.randn(d_model, d_model) * scale
    W_V = np.random.randn(d_model, d_model) * scale
    W_O = np.random.randn(d_model, d_model) * scale

    # ------------------------------------------------------------------
    # 生成输入嵌入
    # ------------------------------------------------------------------
    X = np.random.randn(len(tokens), d_model) * 0.5

    # ------------------------------------------------------------------
    # 计算注意力
    # ------------------------------------------------------------------
    Q = X @ W_Q
    K = X @ W_K
    scores = Q @ K.T / math.sqrt(d_k)

    # 温度缩放
    scaled_scores = scores / temperature
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------
    st.markdown(
        f'<div class="section-title">'
        f'整体注意力权重矩阵&nbsp;&nbsp;{badge(f"{num_heads}-head x {len(tokens)} tokens", "neon")}'
        f'</div>',
        unsafe_allow_html=True,
    )
    fig = render_attention_heatmap(
        attn_weights, tokens,
        title=f"注意力权重 (T={temperature}, {init_method})",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 多头分解
    section_divider()
    section_title(icon="", text="各头注意力分解", size="1.3rem")

    Q_heads = Q.reshape(len(tokens), num_heads, d_k).transpose(1, 0, 2)
    K_heads = K.reshape(len(tokens), num_heads, d_k).transpose(1, 0, 2)

    # 为每个头单独计算注意力权重（供展示和输出投影共用）
    head_attn_weights = []
    for h in range(num_heads):
        h_scores = Q_heads[h] @ K_heads[h].T / math.sqrt(d_k)
        h_scaled = h_scores / temperature
        h_exp = np.exp(h_scaled - np.max(h_scaled, axis=-1, keepdims=True))
        h_weights = h_exp / np.sum(h_exp, axis=-1, keepdims=True)
        head_attn_weights.append(h_weights)

    cols = st.columns(min(num_heads, 4))
    for h in range(num_heads):
        with cols[h % 4]:
            st.caption(f"第 {h+1} 头")
            fig_h = render_attention_heatmap(head_attn_weights[h], tokens, title=f"第 {h+1} 头")
            st.plotly_chart(fig_h, use_container_width=True)

    # 输出投影
    section_divider()
    section_title(icon="", text="输出投影", size="1.3rem")

    V = X @ W_V
    V_heads = V.reshape(len(tokens), num_heads, d_k).transpose(1, 0, 2)

    # 对每个头分别计算注意力输出
    outputs_per_head = []
    for h in range(num_heads):
        h_output = np.matmul(head_attn_weights[h], V_heads[h])
        outputs_per_head.append(h_output)

    attn_output_per_head = np.stack(outputs_per_head, axis=0)  # (num_heads, seq_len, d_k)
    concat_output = attn_output_per_head.transpose(1, 0, 2).reshape(len(tokens), d_model)
    output = concat_output @ W_O

    code_output(f"最终输出形状: {output.shape}")
    st.dataframe(output.round(4), use_container_width=True)

    # ------------------------------------------------------------------
    # 信息面板
    # ------------------------------------------------------------------
    with st.expander("关于 Temperature"):
        st.markdown("""
        Temperature 控制 Softmax 的尖锐程度：
        - **T > 1**：注意力分布更平滑，模型关注更多 token
        - **T < 1**：注意力更集中，模型聚焦少数 token
        - **T = 1**：标准 softmax，无缩放

        公式：$$\\text{softmax}(\\frac{s_i}{T})$$
        """)

with right_col:
    # 右侧参考面板
    st.markdown("""
    <div class="app-right-panel-inner">
        <div class="right-panel-title">📚 参考信息</div>
        <div class="formula-card">
            softmax(QK<sup>T</sup> / √d<sub>k</sub>)
        </div>
        <div class="tip-card">
            <div class="tip-dot" style="background:#f59e0b;"></div>
            <div>
                <div class="tip-title">温度效应</div>
                <div class="tip-body">T > 1 平滑分布，T < 1 集中注意力</div>
            </div>
        </div>
        <div class="tip-card">
            <div class="tip-dot" style="background:#3b82f6;"></div>
            <div>
                <div class="tip-title">注意力解读</div>
                <div class="tip-body">热力图每行表示该 token 对其他 token 的关注程度</div>
            </div>
        </div>
        <div class="tip-card">
            <div class="tip-dot" style="background:#00d4aa;"></div>
            <div>
                <div class="tip-title">初始化对比</div>
                <div class="tip-body">Xavier: 1/√n，Kaiming: √(2/n)，Random: 固定 0.1</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
