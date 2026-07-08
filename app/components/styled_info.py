"""
美化提示框组件.

替代 st.info / st.warning / st.success，提供更一致的样式.
"""

import streamlit as st


def info(title: str, content: str):
    """信息提示框（蓝色）."""
    st.markdown(
        f"""
        <div style="background:#EFF6FF;border-radius:10px;padding:1rem 1.25rem;margin:0.75rem 0;
                     border-left:4px solid #4A90D9;">
            <strong style="color:#1E40AF;">{title}</strong>
            <div style="color:#1A202C;margin-top:0.5rem;line-height:1.6;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning(title: str, content: str):
    """警告提示框（黄色）."""
    st.markdown(
        f"""
        <div style="background:#FFFBEB;border-radius:10px;padding:1rem 1.25rem;margin:0.75rem 0;
                     border-left:4px solid #F59E0B;">
            <strong style="color:#92400E;">{title}</strong>
            <div style="color:#1A202C;margin-top:0.5rem;line-height:1.6;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def success(title: str, content: str):
    """成功提示框（绿色）."""
    st.markdown(
        f"""
        <div style="background:#F0FDF4;border-radius:10px;padding:1rem 1.25rem;margin:0.75rem 0;
                     border-left:4px solid #2EC4B6;">
            <strong style="color:#166534;">{title}</strong>
            <div style="color:#1A202C;margin-top:0.5rem;line-height:1.6;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def error(title: str, content: str):
    """错误提示框（红色）."""
    st.markdown(
        f"""
        <div style="background:#FEF2F2;border-radius:10px;padding:1rem 1.25rem;margin:0.75rem 0;
                     border-left:4px solid #EF4444;">
            <strong style="color:#991B1B;">{title}</strong>
            <div style="color:#1A202C;margin-top:0.5rem;line-height:1.6;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
