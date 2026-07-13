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
            f'<div class="stCard-header">{icon} {title}</div>'
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
        "primary": "background:var(--primary);color:#0a0e1a;",
        "accent": "background:var(--accent);color:white;",
        "outline": "background:transparent;color:var(--primary);border:1px solid var(--primary);",
        "neon": "background:transparent;color:var(--primary);border:1px solid var(--primary);box-shadow:0 0 10px rgba(0,212,255,0.2);",
        "success": "background:var(--success);color:#0a0e1a;",
        "warning": "background:var(--warning);color:#0a0e1a;",
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


def page_title(icon: str, text: str):
    """
    渲染页面标题（替代 st.title，匹配深色科技风主题）.

    Args:
        icon: 标题前图标 emoji
        text: 标题文字
    """
    st.markdown(
        f"""
        <div style="font-size:2rem;font-weight:800;color:var(--text-primary);
                     margin:0.5rem 0 1.5rem 0;
                     text-shadow: 0 0 30px rgba(0,212,255,0.15);">
            {icon} {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(icon: str = "", text: str = "", size: str = "1.3rem"):
    """
    渲染章节标题（替代页面中硬编码的 <div style="font-size:..."> 标题）.

    Args:
        icon: 标题前图标 emoji
        text: 标题文字
        size: 字号，默认 1.3rem
    """
    icon_part = f"{icon} " if icon else ""
    st.markdown(
        f'<div class="section-title" style="font-size:{size};">'
        f"{icon_part}{text}</div>",
        unsafe_allow_html=True,
    )


def section_divider():
    """
    渲染一条章节分隔线.

    Args:
        margin_top: 上边距
        margin_bottom: 下边距
    """
    st.markdown(
        '<hr class="section-divider">',
        unsafe_allow_html=True,
    )


def metric_row(items: list):
    """
    渲染一行指标卡片（用于参数量统计、数据集信息等场景）.

    Args:
        items: 字典列表，每项包含 {label, value, color}
            - label: 标签文字（小字大写）
            - value: 数值/文字（大字加粗）
            - color: 文字颜色，默认 var(--primary)
    """
    html_items = ""
    for item in items:
        color = item.get("color", "var(--primary)")
        html_items += f"""
        <div class="metric-item" style="--metric-color:{color};">
            <div class="metric-item-label">{item['label']}</div>
            <div class="metric-item-value" style="color:var(--metric-color);">{item['value']}</div>
        </div>
        """
    st.markdown(
        f'<div class="metric-item-row">{html_items}</div>',
        unsafe_allow_html=True,
    )
