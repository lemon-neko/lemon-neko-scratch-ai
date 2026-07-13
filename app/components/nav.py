"""
共享导航栏组件.

为所有页面提供统一的顶部导航栏，支持动态高亮当前页面.
"""

import streamlit as st


# 页面 → 路径映射
_NAV_TABS = [
    ("首页", "/"),
    ("项目概览", "/01_项目概览"),
    ("Self-Attention", "/02_SelfAttention"),
    ("注意力热力图", "/03_注意力热力图"),
    ("模型游乐场", "/04_模型游乐场"),
    ("Transformer 块", "/05_Transformer块"),
    ("梯度流分析", "/06_梯度流分析"),
]


def render_nav_bar(active_page: str = "首页"):
    """
    渲染统一的顶部导航栏.

    Args:
        active_page: 当前页面的 tab 名称（必须与 _NAV_TABS 中的名称一致）
    """
    tabs_html = ""
    for name, href in _NAV_TABS:
        active_class = "nav-active" if name == active_page else ""
        tabs_html += (
            f'<a href="{href}" class="top-nav-tab {active_class}">{name}</a>'
        )

    st.markdown(
        f"""
        <div class="top-nav-bar">
            <div class="top-nav-brand">
                <span class="brand-icon">🐱</span>
                <span>AI 教学平台</span>
            </div>
            <div class="top-nav-tabs">
                {tabs_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
