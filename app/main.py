"""
Streamlit 主入口
=================

启动方式：
    streamlit run app/main.py

页面导航通过 app/pages/ 目录下的 Python 文件自动注册.
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

st.markdown(f"<style>{STYLESHEET}</style>", unsafe_allow_html=True)

# ---- Top Navigation Bar ----
st.markdown("""
<div class="top-nav-bar">
    <div class="top-nav-brand">
        <span class="brand-icon">🐱</span>
        <span>AI 全栈可视化教学平台</span>
    </div>
    <div class="top-nav-tabs">
        <a href="/" class="top-nav-tab nav-active">首页</a>
        <a href="/01_项目概览" class="top-nav-tab">项目概览</a>
        <a href="/02_SelfAttention" class="top-nav-tab">Self-Attention</a>
        <a href="/03_注意力热力图" class="top-nav-tab">注意力热力图</a>
        <a href="/04_模型游乐场" class="top-nav-tab">模型游乐场</a>
        <a href="/05_Transformer块" class="top-nav-tab">Transformer 块</a>
        <a href="/06_梯度流分析" class="top-nav-tab">梯度流分析</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Welcome Banner ----
st.markdown("""
<div class="welcome-banner">
    <h1>从零实现 Transformer，用可视化理解 AI</h1>
    <p>纯 NumPy 手写注意力机制、Transformer 块、反向传播，搭配交互式可视化教学平台。<br>调参数、看图表、跑训练 —— 动手学 AI。</p>
</div>
""", unsafe_allow_html=True)

# ---- Stat Metrics Row ----
st.markdown("""
<div class="metric-glow-row">
    <div class="metric-glow">
        <div class="metric-value">6</div>
        <div class="metric-label">教学页面</div>
    </div>
    <div class="metric-glow">
        <div class="metric-value">10K+</div>
        <div class="metric-label">CSS 字符</div>
    </div>
    <div class="metric-glow">
        <div class="metric-value">NumPy</div>
        <div class="metric-label">纯手写实现</div>
    </div>
    <div class="metric-glow">
        <div class="metric-value">MIT</div>
        <div class="metric-label">开源许可</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Footer ----
st.markdown("""
<div class="app-footer">
    基于 "Attention Is All You Need" 论文 · 纯 NumPy 实现 + PyTorch 验证<br>
    作者：lemon-neko · MIT 许可证
</div>
""", unsafe_allow_html=True)
