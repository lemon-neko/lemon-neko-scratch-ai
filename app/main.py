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
    page_title="AI 教学平台",
    page_icon="🐱",
    layout="wide",
)

st.markdown(f"<style>{STYLESHEET}</style>", unsafe_allow_html=True)


# ---- Navigation Helper ----

def render_nav_bar(active_page: str = "首页"):
    """生成顶部导航栏 HTML，支持动态高亮当前页面。

    Parameters
    ----------
    active_page : str
        当前页面的名称，需要与 tabs 列表中的 name 一致。
        其他页面各自传入自己的页面名即可实现导航高亮。
    """
    tabs = [
        ("首页", "/"),
        ("项目概览", "/01_项目概览"),
        ("Self-Attention", "/02_SelfAttention"),
        ("注意力热力图", "/03_注意力热力图"),
        ("模型游乐场", "/04_模型游乐场"),
        ("Transformer 块", "/05_Transformer块"),
        ("梯度流分析", "/06_梯度流分析"),
    ]
    tabs_html = ""
    for name, href in tabs:
        active_class = "nav-active" if name == active_page else ""
        tabs_html += f'<a href="{href}" class="top-nav-tab {active_class}">{name}</a>'

    return f"""
    <div class="top-nav-bar">
        <div class="top-nav-brand">
            <span class="brand-icon">🐱</span>
            <span>AI 教学平台</span>
        </div>
        <div class="top-nav-tabs">
            {tabs_html}
        </div>
    </div>
    """


# ---- Render Navigation ----
st.markdown(render_nav_bar(active_page="首页"), unsafe_allow_html=True)

# ---- Welcome Banner ----
st.markdown("""
<div class="welcome-banner">
    <h1>从零实现 Transformer，用可视化理解 AI</h1>
    <p>纯 NumPy 手写注意力机制、Transformer 块、反向传播，搭配交互式可视化教学平台。<br>调参数、看图表、跑训练 —— 动手学 AI。</p>
</div>
""", unsafe_allow_html=True)

# ---- Metric Cards (6 indicators) ----
st.markdown("""
<div class="metric-glow-row">
    <div class="metric-glow">
        <div class="metric-value">6</div>
        <div class="metric-label">教学模块</div>
    </div>
    <div class="metric-glow">
        <div class="metric-value">30+</div>
        <div class="metric-label">交互式组件</div>
    </div>
    <div class="metric-glow">
        <div class="metric-value">NumPy</div>
        <div class="metric-label">纯手写实现</div>
    </div>
    <div class="metric-glow">
        <div class="metric-value">MIT</div>
        <div class="metric-label">开源许可</div>
    </div>
    <div class="metric-glow">
        <div class="metric-value">Plotly</div>
        <div class="metric-label">可视化</div>
    </div>
    <div class="metric-glow">
        <div class="metric-value">PyTorch</div>
        <div class="metric-label">验证对照</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Page Navigation Cards (5 teaching pages) ----
st.markdown("""
<div class="page-nav-grid">
    <a href="/02_SelfAttention" class="page-nav-card">
        <div class="page-nav-icon">🧠</div>
        <div class="page-nav-title">Self-Attention 详解</div>
        <div class="page-nav-desc">理解注意力机制的核心公式和计算过程</div>
    </a>
    <a href="/03_注意力热力图" class="page-nav-card">
        <div class="page-nav-icon">🌡️</div>
        <div class="page-nav-title">注意力热力图</div>
        <div class="page-nav-desc">可视化注意力权重，观察不同头的关注模式</div>
    </a>
    <a href="/04_模型游乐场" class="page-nav-card">
        <div class="page-nav-icon">🎮</div>
        <div class="page-nav-title">模型游乐场</div>
        <div class="page-nav-desc">配置超参数，训练一个字符级语言模型</div>
    </a>
    <a href="/05_Transformer块" class="page-nav-card">
        <div class="page-nav-icon">🏗️</div>
        <div class="page-nav-title">Transformer 块</div>
        <div class="page-nav-desc">理解编码器/解码器的完整结构</div>
    </a>
    <a href="/06_梯度流分析" class="page-nav-card">
        <div class="page-nav-icon">📊</div>
        <div class="page-nav-title">梯度流分析</div>
        <div class="page-nav-desc">掌握手动反向传播的数学推导</div>
    </a>
</div>
""", unsafe_allow_html=True)

# ---- Footer ----
st.markdown("""
<div class="app-footer">
    基于 "Attention Is All You Need" 论文 · 纯 NumPy 实现 + PyTorch 验证<br>
    作者：lemon-neko · MIT 许可证
</div>
""", unsafe_allow_html=True)
