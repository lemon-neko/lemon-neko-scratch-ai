"""
通用卡片组件.

提供可复用的卡片、徽章、信息面板等 UI 元素.
"""

from typing import Optional

import streamlit as st


def card(title: Optional[str] = None, icon: str = "", children: str = ""):
    """
    渲染一个带可选标题的卡片容器.

    Args:
        title: 卡片标题
        icon: 标题前图标
        children: 卡片内容 HTML
    """
    header = ""
    if title:
        header = f"<div style='font-weight:700;font-size:1rem;color:#1A202C;margin-bottom:0.75rem;'>{icon} {title}</div>"
    st.markdown(
        f'<div class="stCard">{header}{children}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, variant: str = "primary"):
    """
    渲染一个徽章/标签.

    Args:
        text: 徽章文字
        variant: "primary" | "accent" | "outline"
    """
    cls_map = {
        "primary": "badge",
        "accent": "badge badge-accent",
        "outline": "badge badge-outline",
    }
    cls = cls_map.get(variant, "badge")
    return f'<span class="{cls}">{text}</span>'


def info_panel(content: str, title: str = ""):
    """
    渲染一个信息面板（带左边框的高亮区块）.

    Args:
        content: 面板内容 HTML
        title: 可选标题
    """
    header = f"<strong>{title}</strong><br>" if title else ""
    st.markdown(
        f'<div class="info-panel">{header}{content}</div>',
        unsafe_allow_html=True,
    )


def tech_card(icon: str, name: str, desc: str):
    """
    渲染一张技术栈卡片.

    Args:
        icon: 图标 emoji
        name: 技术名称
        desc: 描述文字
    """
    st.markdown(f"""
    <div class="tech-card">
        <span class="tech-icon">{icon}</span>
        <div class="tech-name">{name}</div>
        <div class="tech-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def route_card(step: int, title: str, desc: str):
    """
    渲染一条学习路线卡片.

    Args:
        step: 步骤编号
        title: 标题
        desc: 描述
    """
    st.markdown(f"""
    <div class="route-card">
        <span class="route-num">{step}</span>
        <div class="route-content">
            <div class="route-title">{title}</div>
            <div class="route-desc">{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def comparison_card(title: str, icon: str, items: list):
    """
    渲染一张对比卡片（用于 Encoder vs Decoder 等场景）.

    Args:
        title: 卡片标题
        icon: 图标
        items: 列表项字符串列表
    """
    li = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(f"""
    <div class="comparison-card">
        <h4>{icon} {title}</h4>
        <ul>{li}</ul>
    </div>
    """, unsafe_allow_html=True)


def code_output(text: str):
    """
    渲染一个代码输出块.

    Args:
        text: 代码文本
    """
    st.markdown(
        f'<div class="code-output">{text}</div>',
        unsafe_allow_html=True,
    )


def gen_text_box(label: str, content: str):
    """
    渲染一个文本生成结果框.

    Args:
        label: 标签
        content: 内容
    """
    st.markdown(f"""
    <div class="gen-text-box">
        <div class="gen-label">{label}</div>
        <div class="gen-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)
