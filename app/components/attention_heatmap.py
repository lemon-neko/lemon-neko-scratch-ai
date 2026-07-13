"""
注意力热力图 Plotly 组件
==========================

提供交互式注意力权重可视化，支持单头和多头子图布局.
适配深色科技风主题.
"""

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go

# 自定义 teal-cyan 色阶，与暗色科技风主题匹配
TEAL_HEATMAP_COLORSCALE = [
    [0.0, "#060a14"],     # 最深背景
    [0.15, "#0a1628"],
    [0.3, "#0d3d5a"],
    [0.45, "#0d5f7a"],
    [0.6, "#0d8a7a"],
    [0.75, "#00d4aa"],
    [0.9, "#33e0bf"],
    [1.0, "#ffe040"],      # 最亮（高注意力）
]


def _apply_dark_theme(fig: go.Figure):
    """Apply dark theme styling to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        title_font=dict(color="#E2E8F0"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#E2E8F0"),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#E2E8F0"),
    )
    fig.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color="#94A3B8"),
            title_font=dict(color="#E2E8F0"),
            bgcolor="rgba(0,0,0,0)",
            outlinewidth=0,
            thickness=15,
            len=0.75,
        )
    )
    return fig


def render_attention_heatmap(
    weights: np.ndarray,
    tokens: List[str],
    title: str = "注意力权重",
    head_names: Optional[List[str]] = None,
    temperature: Optional[float] = None,
) -> go.Figure:
    """
    渲染注意力权重热力图.

    支持两种模式：
    - 单头：直接显示 (seq_len, seq_len) 矩阵
    - 多头：自动拆分为 num_heads 个子图

    Args:
        weights: 注意力权重矩阵
            - 单头: (seq_len, seq_len)
            - 多头: (num_heads, seq_len, seq_len)
        tokens: token 标签列表
        title: 图表标题
        head_names: 多头时的头名称列表（如 "Head 1", "Head 2"）
        temperature: 温度系数（>1 使注意力更平滑，<1 使其更尖锐）

    Returns:
        Plotly Figure 对象（深色主题）
    """
    if weights.ndim == 2:
        # 单头模式
        weights = weights[np.newaxis, ...]

    num_heads, seq_len, _ = weights.shape
    if head_names is None:
        head_names = [f"第 {i+1} 头" for i in range(num_heads)]

    # 应用温度系数
    if temperature is not None and temperature != 1.0:
        scaled = weights / temperature
        # 重新 softmax
        exp_s = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

    if num_heads == 1:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=weights[0],
            x=tokens,
            y=tokens,
            colorscale=TEAL_HEATMAP_COLORSCALE,
            hovertemplate="查询: %{y}<br>键: %{x}<br>权重: %{z:.4f}<extra></extra>",
            colorbar=dict(title="权重", tickformat=".2f"),
        ))
        fig.update_layout(
            title=title,
            xaxis_title="键 (token)",
            yaxis_title="查询 (token)",
            height=400,
            width=500,
        )
    else:
        # 多图布局
        cols = min(num_heads, 4)
        rows = (num_heads + cols - 1) // cols
        fig = go.Figure()

        for i in range(num_heads):
            fig.add_trace(go.Heatmap(
                z=weights[i],
                x=tokens,
                y=tokens,
                colorscale=TEAL_HEATMAP_COLORSCALE,
                name=head_names[i],
                hovertemplate=(
                    f"{head_names[i]}<br>查询: %{{y}}<br>"
                    f"键: %{{x}}<br>权重: %{{z:.4f}}<extra></extra>"
                ),
                visible=i == 0,
            ))

        fig.update_layout(
            title=title,
            xaxis_title="键 (token)",
            yaxis_title="查询 (token)",
            height=rows * 300 + 100,
            width=cols * 250 + 50,
            legend_title="注意力头",
        )

        # 添加头选择下拉框
        if num_heads > 1:
            buttons = []
            # "所有头" 按钮
            buttons.append({
                "label": "所有头",
                "method": "update",
                "args": [{"visible": [True] * num_heads}],
            })
            # 每个头的按钮
            for i in range(num_heads):
                buttons.append({
                    "label": head_names[i],
                    "method": "update",
                    "args": [{"visible": [j == i for j in range(num_heads)]}],
                })
            fig.update_layout(
                updatemenus=[{
                    "type": "dropdown",
                    "buttons": buttons,
                    "showactive": True,
                    "direction": "down",
                    "font": dict(color="#E2E8F0"),
                    "bgcolor": "#151B2B",
                    "bordercolor": "rgba(255,255,255,0.08)",
                }],
            )

    return _apply_dark_theme(fig)


def render_attention_comparison(
    weights_list: List[np.ndarray],
    tokens: List[str],
    titles: Optional[List[str]] = None,
) -> go.Figure:
    """
    并排比较多个注意力权重矩阵.

    Args:
        weights_list: 多个权重矩阵列表，每个 (seq_len, seq_len) 或 (num_heads, seq_len, seq_len)
        tokens: token 标签
        titles: 每个子图的标题

    Returns:
        Plotly Figure 对象（深色主题）
    """
    if titles is None:
        titles = [f"权重组 {i+1}" for i in range(len(weights_list))]

    fig = go.Figure()

    for i, w in enumerate(weights_list):
        if w.ndim == 3:
            # 取平均注意力
            avg_w = np.mean(w, axis=0)
        else:
            avg_w = w

        fig.add_trace(go.Heatmap(
            z=avg_w,
            x=tokens,
            y=tokens,
            colorscale="Viridis",
            name=titles[i],
            hovertemplate=(
                f"{titles[i]}<br>查询: %{{y}}<br>"
                f"键: %{{x}}<br>权重: %{{z:.4f}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="注意力对比",
        opacity=0.7,
        xaxis_title="键 (token)",
        yaxis_title="查询 (token)",
        height=400,
        width=500,
        legend_title="对比组",
    )

    return _apply_dark_theme(fig)
