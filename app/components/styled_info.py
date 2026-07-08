"""
美化提示框组件.

替代 st.info / st.warning / st.success，提供更一致的样式.
适配深色科技风主题.
"""

import streamlit as st


def info(title: str, content: str):
    """信息提示框（霓虹青左边框）."""
    st.markdown(
        f"""
        <div style="background:rgba(0,212,255,0.06);border-radius:10px;padding:1rem 1.25rem;margin:0.75rem 0;
                     border-left:4px solid var(--primary);">
            <strong style="color:var(--primary-light);">{title}</strong>
            <div style="color:var(--text-primary);margin-top:0.5rem;line-height:1.6;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning(title: str, content: str):
    """警告提示框（霓虹琥珀左边框）."""
    st.markdown(
        f"""
        <div style="background:rgba(255,171,0,0.06);border-radius:10px;padding:1rem 1.25rem;margin:0.75rem 0;
                     border-left:4px solid var(--warning);">
            <strong style="color:var(--warning);">{title}</strong>
            <div style="color:var(--text-primary);margin-top:0.5rem;line-height:1.6;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def success(title: str, content: str):
    """成功提示框（霓虹绿左边框）."""
    st.markdown(
        f"""
        <div style="background:rgba(0,230,118,0.06);border-radius:10px;padding:1rem 1.25rem;margin:0.75rem 0;
                     border-left:4px solid var(--success);">
            <strong style="color:var(--success);">{title}</strong>
            <div style="color:var(--text-primary);margin-top:0.5rem;line-height:1.6;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def error(title: str, content: str):
    """错误提示框（霓虹红左边框）."""
    st.markdown(
        f"""
        <div style="background:rgba(255,82,82,0.06);border-radius:10px;padding:1rem 1.25rem;margin:0.75rem 0;
                     border-left:4px solid var(--error);">
            <strong style="color:var(--error);">{title}</strong>
            <div style="color:var(--text-primary);margin-top:0.5rem;line-height:1.6;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
