"""
Home 页面 — 项目概览与学习路线
"""

import streamlit as st
from components.card import badge, info_panel, route_card, tech_card

st.set_page_config(page_title="项目概览", layout="wide")

st.title("📖 项目概览")

st.markdown("""
本平台旨在通过**交互式可视化**的方式，带你从零理解 Transformer 的核心原理。
所有内容均使用纯 NumPy 实现，并与 PyTorch 进行对比验证。
""")

# ---- 学习路线 ----
st.header("🗺️ 学习路线")

st.markdown(
    '<p style="color:#718096;font-size:0.95rem;margin-bottom:1rem;">'
    "建议按以下顺序学习，每页都建立在上一页的基础上："
    "</p>",
    unsafe_allow_html=True,
)

route_card(1, "Self-Attention 详解", "理解注意力公式、Q/K/V 投影、多头机制")
route_card(2, "注意力热力图", "可视化注意力权重，观察不同头的关注模式")
route_card(3, "模型游乐场", "配置超参数，亲手训练一个字符级语言模型")
route_card(4, "Transformer 块", "理解编码器/解码器的完整结构")
route_card(5, "梯度流分析", "掌握手动反向传播的数学推导")

# ---- 核心公式 ----
st.header("🧮 核心公式")

with st.expander(f"{badge('Step 1')} Scaled Dot-Product Attention"):
    st.latex(
        r"""\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V"""
    )

with st.expander(f"{badge('Step 2')} Multi-Head Attention"):
    st.latex(
        r"""\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O"""
    )
    st.latex(
        r"""\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)"""
    )

with st.expander(f"{badge('Step 3')} Transformer Encoder Block"):
    st.latex(
        r"""\begin{aligned}
    \text{sublayer}_1(L) &= \text{LayerNorm}(L + \text{MultiHead}(L)) \\
    \text{sublayer}_2(L) &= \text{LayerNorm}(L + \text{FFN}(\text{sublayer}_1(L)))
    \end{aligned}"""
    )

# ---- 技术栈 ----
st.header("🛠️ 技术栈")

st.markdown('<div style="display:flex;gap:1rem;flex-wrap:wrap;">', unsafe_allow_html=True)
tech_card(
    "NumPy",
    "**NumPy** — 从零实现所有矩阵运算，无深度学习框架依赖",
)
tech_card(
    "PyTorch",
    "**PyTorch** — 作为参考实现，验证 NumPy 正确性",
)
tech_card(
    "Streamlit + Plotly",
    "**Streamlit + Plotly** — 交互式可视化，零 JavaScript",
)
st.markdown("</div>", unsafe_allow_html=True)

# ---- 前置知识 ----
st.header("📚 前置知识")

info_panel(
    title="📌 阅读建议",
    content="""
    <p style="margin:0 0 0.75rem 0;">
    阅读本平台的代码前，建议你先了解：
    </p>
    <ul style="margin:0 0 0.75rem 0;padding-left:1.25rem;">
        <li>线性代数：矩阵乘法、转置、特征值</li>
        <li>微积分：链式法则、偏导数</li>
        <li>深度学习基础：前向传播、反向传播、Softmax</li>
    </ul>
    <p style="margin:0 0 0.5rem 0;">如果你对这些概念还不太熟悉，可以参考：</p>
    <ul style="margin:0;padding-left:1.25rem;">
        <li><a href="https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab" target="_blank" style="color:#4A90D9;text-decoration:none;">3Blue1Brown — 线性代数的本质</a></li>
        <li><a href="https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr" target="_blank" style="color:#4A90D9;text-decoration:none;">3Blue1Brown — 神经网络的几何意义</a></li>
    </ul>
    """,
)
