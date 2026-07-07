"""
Home 页面 — 项目概览与学习路线
"""

import streamlit as st

st.title("📖 项目概览")

st.markdown("""
本平台旨在通过**交互式可视化**的方式，带你从零理解 Transformer 的核心原理。
所有内容均使用纯 NumPy 实现，并与 PyTorch 进行对比验证。
""")

# ---- 学习路线 ----
st.header("🗺️ 学习路线")

st.markdown("""
建议按以下顺序学习，每页都建立在上一页的基础上：

1. **Self-Attention 详解** — 理解注意力公式、Q/K/V 投影、多头机制
2. **注意力热力图** — 可视化注意力权重，观察不同头的关注模式
3. **模型游乐场** — 配置超参数，亲手训练一个字符级语言模型
4. **Transformer 块** — 理解编码器/解码器的完整结构
5. **梯度流分析** — 掌握手动反向传播的数学推导
""")

# ---- 核心公式 ----
st.header("🧮 核心公式")

with st.expander("Scaled Dot-Product Attention"):
    st.latex(r"""\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V""")

with st.expander("Multi-Head Attention"):
    st.latex(r"""\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O""")
    st.latex(r"""\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)""")

with st.expander("Transformer Encoder Block"):
    st.latex(r"""\begin{aligned}
    \text{sublayer}_1(L) &= \text{LayerNorm}(L + \text{MultiHead}(L)) \\
    \text{sublayer}_2(L) &= \text{LayerNorm}(L + \text{FFN}(\text{sublayer}_1(L)))
    \end{aligned}""")

# ---- 技术栈 ----
st.header("🛠️ 技术栈")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**NumPy** — 从零实现所有矩阵运算，无深度学习框架依赖")
with col2:
    st.info("**PyTorch** — 作为参考实现，验证 NumPy 正确性")
with col3:
    st.info("**Streamlit + Plotly** — 交互式可视化，零 JavaScript")

# ---- 前置知识 ----
st.header("📚 前置知识")
st.markdown("""
阅读本平台的代码前，建议你先了解：
- 线性代数：矩阵乘法、转置、特征值
- 微积分：链式法则、偏导数
- 深度学习基础：前向传播、反向传播、Softmax

如果你对这些概念还不太熟悉，可以参考：
- [3Blue1Brown — 线性代数的本质](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown — 神经网络的几何意义](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
""")
