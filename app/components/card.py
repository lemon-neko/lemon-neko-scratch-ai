"""
通用卡片组件.

提供可复用的卡片、徽章、信息面板等 UI 元素.
适配深色科技风主题.
"""

from typing import Optional

import streamlit as st


def card(
    title: Optional[str] = None,
    icon: str = "",
    children: str = "",
    glow: bool = True,
):
    """
    渲染一个带可选标题的卡片容器.

    Args:
        title: 卡片标题
        icon: 标题前图标
        children: 卡片内容 HTML
        glow: 是否添加霓虹发光边框
    """
    header = ""
    if title:
        header = (
            f"<div style='font-weight:700;font-size:1rem;"
            f"color:var(--text-primary);margin-bottom:0.75rem;'>"
            f"{icon} {title}</div>"
        )
    glow_class = " glow-border" if glow else ""
    st.markdown(
        f'<div class="stCard{glow_class}">{header}{children}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, variant: str = "primary"):
    """
    渲染一个徽章/标签.

    Args:
        text: 徽章文字
        variant: "primary" | "accent" | "outline" | "neon" | "success" | "warning" | "error"
    """
    cls_map = {
        "primary": "badge",
        "accent": "badge badge-accent",
        "outline": "badge badge-outline",
        "neon": "badge badge-neon",
        "success": "badge",
        "warning": "badge",
        "error": "badge",
    }
    style_map = {
        "primary": "background:var(--primary);color:#0B0F19;",
        "accent": "background:var(--accent);color:white;",
        "outline": "background:transparent;color:var(--primary);border:1px solid var(--primary);",
        "neon": "background:transparent;color:var(--primary);border:1px solid var(--primary);box-shadow:0 0 10px rgba(0,212,255,0.2);",
        "success": "background:var(--success);color:#0B0F19;",
        "warning": "background:var(--warning);color:#0B0F19;",
        "error": "background:var(--error);color:white;",
    }
    cls = cls_map.get(variant, "badge")
    style = style_map.get(variant, style_map["primary"])
    return f'<span class="{cls}" style="{style}">{text}</span>'


def info_panel(content: str, title: str = ""):
    """
    渲染一个信息面板（带左边框的高亮区块）.

    Args:
        content: 面板内容 HTML
        title: 可选标题
    """
    header = f"<strong style='color:var(--primary);'>{title}</strong><br>" if title else ""
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
