"""
统一侧边栏组件.

提供带卡片样式的侧边栏配置区域，支持分组和标题.
"""

import streamlit as st


def render_config_card(title: str = "配置", icon: str = "⚙️"):
    """
    渲染一个侧边栏配置卡片容器.

    Args:
        title: 卡片标题
        icon: 标题前的图标
    """
    st.markdown(f"""
    <div class="config-card">
        <h4>{icon} {title}</h4>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_header(title: str = "配置", icon: str = "⚙️"):
    """
    在侧边栏顶部渲染统一标题.

    Args:
        title: 标题文字
        icon: 图标 emoji
    """
    st.sidebar.markdown(f"""
    <div style="margin-bottom:1rem;">
        <span style="font-size:1.5rem;">{icon}</span>
        <span style="font-size:1.1rem;font-weight:700;color:#1A202C;margin-left:0.5rem;">{title}</span>
    </div>
    <hr style="border:none;border-top:1px solid #E2E8F0;margin:0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)


def render_sidebar_divider():
    """在侧边栏中渲染一条分隔线."""
    st.sidebar.markdown(
        '<hr style="border:none;border-top:1px solid #E2E8F0;margin:0.75rem 0;">',
        unsafe_allow_html=True,
    )


def render_sidebar_metric(label: str, value: str):
    """在侧边栏中渲染一行指标."""
    st.sidebar.caption(f"{label}: **{value}**")
