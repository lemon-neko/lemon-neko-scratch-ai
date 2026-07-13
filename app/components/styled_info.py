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
        <div class="info-box">
            <strong class="info-box-title">{title}</strong>
            <div class="info-box-body">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def warning(title: str, content: str):
    """警告提示框（霓虹琥珀左边框）."""
    st.markdown(
        f"""
        <div class="info-box info-box--warning">
            <strong class="info-box-title">{title}</strong>
            <div class="info-box-body">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def success(title: str, content: str):
    """成功提示框（霓虹绿左边框）."""
    st.markdown(
        f"""
        <div class="info-box info-box--success">
            <strong class="info-box-title">{title}</strong>
            <div class="info-box-body">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def error(title: str, content: str):
    """错误提示框（霓虹红左边框）."""
    st.markdown(
        f"""
        <div class="info-box info-box--error">
            <strong class="info-box-title">{title}</strong>
            <div class="info-box-body">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
