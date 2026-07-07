"""
注意力热力图 Plotly 组件
==========================

提供交互式注意力权重可视化，支持单头和多头子图布局。
"""

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go


def render_attention_heatmap(
    weights: np.ndarray,
    tokens: List[str],
    title: str = "Attention Weights",
    head_names: Optional[List[str]] = None,
    temperature: Optional[float] = None,
) -> go.Figure:
    """
    渲染注意力权重热力图。

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
        Plotly Figure 对象
    """
    if weights.ndim == 2:
        # 单头模式
        weights = weights[np.newaxis, ...]

    num_heads, seq_len, _ = weights.shape
    if head_names is None:
        head_names = [f"Head {i+1}" for i in range(num_heads)]

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
            colorscale="RdYlGn_r",
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>",
            colorbar=dict(title="Weight", tickformat=".2f"),
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Key (token)",
            yaxis_title="Query (token)",
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
                colorscale="RdYlGn_r",
                name=head_names[i],
                hovertemplate=(
                    f"{head_names[i]}<br>Query: %{{y}}<br>"
                    f"Key: %{{x}}<br>Weight: %{{z:.4f}}<extra></extra>"
                ),
                visible=i == 0,
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Key (token)",
            yaxis_title="Query (token)",
            height=rows * 300 + 100,
            width=cols * 250 + 50,
            legend_title="Heads",
            barmode="relative",
        )

        # 添加头选择下拉框
        if num_heads > 1:
            fig.update_layout(
                updatemenus=[{
                    "type": "dropdown",
                    "buttons": [
                        {
                            "label": "All Heads",
                            "method": "update",
                            "args": [{"visible": [True] * num_heads}],
                        },
                        *[
                            {
                                "label": head_names[i],
                                "method": "update",
                                "args": [{"visible": [j == i for j in range(num_heads)]}],
                            }
                            for i in range(num_heads)
                        ],
                    ],
                    "showlabel": True,
                    "direction": "down",
                }],
            )

    return fig


def render_attention_comparison(
    weights_list: List[np.ndarray],
    tokens: List[str],
    titles: Optional[List[str]] = None,
) -> go.Figure:
    """
    并排比较多个注意力权重矩阵。

    Args:
        weights_list: 多个权重矩阵列表，每个 (seq_len, seq_len) 或 (num_heads, seq_len, seq_len)
        tokens: token 标签
        titles: 每个子图的标题

    Returns:
        Plotly Figure 对象
    """
    if titles is None:
        titles = [f"Weights {i+1}" for i in range(len(weights_list))]

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
                f"{titles[i]}<br>Query: %{{y}}<br>"
                f"Key: %{{x}}<br>Weight: %{{z:.4f}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Attention Comparison",
        barmode="overlay",
        opacity=0.7,
        xaxis_title="Key (token)",
        yaxis_title="Query (token)",
        height=400,
        width=500,
        legend_title="Comparison",
    )

    return fig
