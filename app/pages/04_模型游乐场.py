"""Train and sample a small NumPy decoder-only language model."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.card import gen_text_box, page_title, section_divider, section_title
from components.nav import render_nav_bar
from components.styled_info import info, success
from src.minigpt import CharTokenizer, MiniGPT, make_language_model_batches


DEFAULT_TEXT = "我爱看猫。猫很可爱。\n我喜欢学习 Transformer。\nhello transformer! "


def loss_chart(losses: list[float]) -> go.Figure:
    chart = go.Figure(go.Scatter(y=losses, mode="lines", line={"color": "#00d4aa", "width": 2}))
    chart.update_layout(
        title="真实交叉熵损失", xaxis_title="训练步数", yaxis_title="Loss", height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#E2E8F0"},
    )
    return chart


st.set_page_config(page_title="Mini-GPT 游乐场", layout="wide", page_icon="🐱")
render_nav_bar(active_page="模型游乐场")
page_title("🎮", "Mini-GPT 游乐场")
st.markdown("用纯 NumPy 训练一个真正的 Decoder-only Transformer：因果掩码、交叉熵、解析反向传播和采样生成都在这里发生。")

left, main, right = st.columns([1, 4, 1])
with left:
    st.markdown("#### 模型参数")
    d_model = st.select_slider("d_model", options=[16, 24, 32, 48, 64], value=32)
    head_options = [head for head in (1, 2, 3, 4, 6, 8) if d_model % head == 0]
    num_heads = st.select_slider("注意力头数", options=head_options, value=next((h for h in (4, 2, 1) if h in head_options)))
    num_layers = st.slider("Decoder 层数", 1, 3, 1)
    block_size = st.slider("上下文长度", 4, 32, 12, step=2)
    learning_rate = st.slider("学习率", 0.01, 0.20, 0.08, 0.01)
    steps = st.slider("训练步数", 10, 300, 100, 10)

with main:
    section_title(icon="📚", text="训练语料", size="1.3rem")
    corpus = st.text_area("输入文本（字符级分词）", DEFAULT_TEXT, height=130)
    if corpus:
        tokenizer_preview = CharTokenizer.from_text(corpus)
        info("数据集信息", f"词表大小：**{tokenizer_preview.vocab_size}**｜字符数：**{len(corpus)}**")

    section_divider()
    if st.button("▶ 训练 Mini-GPT", type="primary"):
        if len(corpus) <= block_size:
            st.error("训练文本必须长于上下文长度。")
        else:
            tokenizer = CharTokenizer.from_text(corpus)
            token_ids = tokenizer.encode(corpus)
            model = MiniGPT(tokenizer.vocab_size, d_model=d_model, num_heads=num_heads, d_ff=d_model * 2,
                            num_layers=num_layers, block_size=block_size)
            rng = np.random.default_rng(17)
            losses: list[float] = []
            progress = st.progress(0)
            for step in range(steps):
                inputs, targets = make_language_model_batches(token_ids, block_size, min(8, len(token_ids) - block_size), rng)
                losses.append(model.train_step(inputs, targets, learning_rate))
                progress.progress((step + 1) / steps)
            st.session_state.minigpt = model
            st.session_state.minigpt_tokenizer = tokenizer
            st.session_state.minigpt_losses = losses
            success("训练完成", f"最终 loss：{losses[-1]:.3f}（起始：{losses[0]:.3f}）")

    losses = st.session_state.get("minigpt_losses", [])
    if losses:
        st.plotly_chart(loss_chart(losses), use_container_width=True)
        section_title(icon="✍️", text="用训练好的模型生成", size="1.3rem")
        prompt = st.text_input("提示文本", "我")
        temperature = st.slider("采样温度", 0.2, 1.5, 0.8, 0.1)
        if st.button("生成文本"):
            tokenizer = st.session_state.minigpt_tokenizer
            try:
                output = st.session_state.minigpt.generate(tokenizer.encode(prompt), 40, temperature, seed=None)
                gen_text_box("生成结果", tokenizer.decode(output))
            except ValueError as error:
                st.error(f"无法生成：{error}")
    else:
        st.caption("训练后会显示真实损失曲线和模型采样结果。")

with right:
    st.markdown("#### 这次真正训练了什么？")
    st.info("输入字符 → Token Embedding + Position Embedding → 因果多头注意力 → FFN → 下一个字符概率")
    st.caption("未来字符在注意力分数中被遮蔽；训练更新所有嵌入、注意力、前馈和输出层参数，而不是随机扰动。")
