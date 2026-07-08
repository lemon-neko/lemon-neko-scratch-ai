"""
Streamlit 主入口
=================

启动方式：
    streamlit run app/main.py

页面导航通过 app/pages/ 目录下的 Python 文件自动注册。
"""

import os
import sys

import streamlit as st

# 确保 src/ 和 app/ 下的模块可以被导入
# 本地开发时 sys.path 已有仓库根，Cloud 上需要显式添加
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from styles import STYLESHEET  # noqa: E402

st.set_page_config(
    page_title="AI 全栈可视化教学平台",
    page_icon="🐱",
    layout="wide",
)

st.title("🐱 AI 全栈可视化教学平台")

st.markdown(f"<style>{STYLESHEET}</style>", unsafe_allow_html=True)

# ---- Welcome Banner ----
st.markdown("""
<div class="welcome-banner">
    <h1>从零实现 Transformer，用可视化理解 AI</h1>
    <p>纯 NumPy 手写注意力机制、Transformer 块、反向传播，搭配交互式可视化教学平台。<br>调参数、看图表、跑训练 —— 动手学 AI。</p>
</div>
""", unsafe_allow_html=True)

# ---- Quick Start ----
st.markdown("### 🚀 快速开始")

page_cards = [
    ("📖 首页", "项目概览、目录结构、学习路线", "01_home"),
    ("🔬 Self-Attention 详解", "交互式逐步讲解注意力机制", "02_self_attention"),
    ("🔍 注意力热力图", "可视化注意力权重分布", "03_attention_visualizer"),
    ("🎮 模型游乐场", "配置超参数，训练玩具语言模型", "04_model_playground"),
    ("🧱 Transformer 块", "编码器/解码器块的结构可视化", "05_transformer_block"),
    ("📉 梯度流分析", "手动反向传播与梯度范数展示", "06_gradient_flow"),
]

for icon, desc, page in page_cards:
    st.markdown(
        f'<a href="/{page}" class="page-nav-card" style="text-decoration:none;color:inherit;">'
        f'<span class="nav-icon">{icon}</span>'
        f'<div class="nav-title">{page.replace("_", " ").title()}</div>'
        f'<div class="nav-desc">{desc}</div>'
        f'</a>',
        unsafe_allow_html=True,
    )

st.markdown("")  # spacer

# ---- How to Run ----
st.markdown("### 💻 运行本地")
st.markdown("""
```bash
pip install -r requirements-app.txt
streamlit run app/main.py
```
""")

# ---- Project Structure ----
st.markdown("### 📁 项目结构")
st.markdown("""
```
lemon-neko-scratch-ai/
├── src/                  ← 从零实现的 ML 模块
│   ├── attention.py      ← Self-Attention 核心
│   └── layers.py         ← LayerNorm, FFN, 位置编码
├── app/                  ← Streamlit 可视化教学应用
│   ├── main.py           ← 入口
│   ├── pages/            ← 各教学页面
│   └── components/       ← 可复用可视化组件
├── notebooks/            ← Jupyter 教程笔记本
├── examples/             ← 独立可运行脚本
├── tests/                ← 单元测试
├── notes/                ← 学习笔记
├── experiments/          ← 实验配置与结果
├── data/                 ← 小型示例数据
├── assets/               ← 静态资源
├── tools/                ← 开发与部署脚本
├── docs/                 ← 详细文档
└── .github/workflows/    ← CI
```
""")

# ---- Footer ----
st.markdown("""
<div class="app-footer">
    基于 "Attention Is All You Need" 论文 · 纯 NumPy 实现 + PyTorch 验证<br>
    作者：lemon-neko · MIT 许可证
</div>
""", unsafe_allow_html=True)
