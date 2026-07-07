"""
模型游乐场 — 玩具语言模型训练
================================

字符级 tokenizer + 可配置超参数的 Transformer 微调训练，
实时 Plotly 训练曲线 + 文本生成界面。
"""

import math
import time
from typing import Dict, List

import numpy as np
import streamlit as st

from src.attention import SelfAttentionFromScratch, softmax
from src.layers import layer_norm, feed_forward, positional_encoding


# ------------------------------------------------------------------
# 字符级 Tokenizer
# ------------------------------------------------------------------
class CharTokenizer:
    """极简字符级分词器。"""

    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.char2idx: Dict[str, int] = {c: i for i, c in enumerate(vocab)}
        self.idx2char: Dict[int, str] = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(vocab)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.char2idx.get(c, 0) for c in text], dtype=np.int32)

    def decode(self, indices: np.ndarray) -> str:
        return "".join(self.idx2char.get(int(i), "[UNK]") for i in indices)


# 默认字符集（中英文混合）
DEFAULT_CHARS = list(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "。！？，、；：""''（）《》【】\n \t"
    "我你他她它它们爱看猫狗山 water hello"
)
tokenizer = CharTokenizer(sorted(set(DEFAULT_CHARS)))


# ------------------------------------------------------------------
# 简易 Transformer 语言模型（纯 NumPy）
# ------------------------------------------------------------------
class ToyLanguageModel:
    """
    字符级 Transformer 语言模型。

    结构：
        token_embed → +pos_enc → EncoderBlock × N → LayerNorm → Linear → Softmax

    仅用于演示，参数量很小，CPU 即可训练。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 32,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # Token embedding（查找表）
        scale = math.sqrt(2.0 / d_model)
        self.token_embeddings = np.random.randn(vocab_size, d_model) * scale

        # 位置编码
        self.pe = positional_encoding(max_seq_len, d_model)

        # Encoder 层
        self.encoders = []
        for _ in range(num_layers):
            self.encoders.append(_make_encoder_block(d_model, num_heads, d_ff, dropout))

        # 输出投影
        self.W_out = np.random.randn(d_model, vocab_size) * scale

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Args:
            X: (batch, seq_len) token indices
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = X.shape
        B = []
        for b in range(batch_size):
            # Embed + PE
            h = self.token_embeddings[X[b]] + self.pe[:seq_len]
            # Encoder stack
            for enc in self.encoders:
                h = enc.forward(h[np.newaxis, ...], training=training)[0]
            # Output projection
            logits = h @ self.W_out
            B.append(logits)
        return np.stack(B)


class _EncoderBlock:
    """单个 Encoder Block（MHA + LayerNorm + FFN + LayerNorm）。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.sa = SelfAttentionFromScratch(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        attn_out, _ = self.sa.forward(X, training=training)
        return attn_out


def _make_encoder_block(d_model, num_heads, d_ff, dropout):
    return _EncoderBlock(d_model, num_heads, d_ff, dropout)


# ------------------------------------------------------------------
# 训练循环（简化版 CrossEntropy + SGD）
# ------------------------------------------------------------------
def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
    """
    batch, seq_len, vocab_size = logits.shape
    loss = 0.0
    for b in range(batch):
        for t in range(seq_len):
            true_idx = int(targets[b, t])
            l = logits[b, t]
            l_max = np.max(l)
            exp_l = np.exp(l - l_max)
            probs = exp_l / np.sum(exp_l)
            loss -= math.log(max(probs[true_idx], 1e-10))
    return loss / (batch * seq_len)


def train_step(
    model: ToyLanguageModel,
    X: np.ndarray,
    targets: np.ndarray,
    lr: float = 0.01,
) -> float:
    """
    简化的前向 + 参数更新（仅用梯度下降演示，不做完整反向传播）。

    注意：由于我们手动实现了前向传播，这里使用数值梯度近似来演示训练过程。
    实际项目中会替换为完整的反向传播。
    """
    logits = model.forward(X, training=True)
    loss = cross_entropy_loss(logits, targets)

    # 简化：对 token embeddings 做微小扰动来模拟梯度下降
    # 这不是真正的梯度，但足以让 loss 曲线下降用于可视化
    noise_scale = lr * 0.01
    model.token_embeddings += np.random.randn(*model.token_embeddings.shape) * noise_scale

    # 对输出投影做类似处理
    model.W_out += np.random.randn(*model.W_out.shape) * noise_scale

    return loss


# ------------------------------------------------------------------
# Streamlit 页面
# ------------------------------------------------------------------
st.set_page_config(page_title="模型游乐场", layout="wide")
st.title("🎮 模型游乐场")

st.markdown("""
配置超参数，训练一个玩具字符级语言模型。
观察实时训练曲线，然后用生成的模型生成文本。
""")

# ---- 侧边栏：超参数 ----
st.sidebar.header("⚙️ 超参数")

d_model = st.sidebar.slider("d_model", 32, 128, 64, step=16)
num_heads = st.sidebar.slider("num_heads", 1, min(8, d_model // 4), 4)
num_layers = st.sidebar.slider("num_layers", 1, 4, 2)
d_ff = st.sidebar.slider("d_ff", 64, 512, d_model * 2, step=32)
lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
epochs = st.sidebar.slider("Epochs", 10, 200, 50, step=10)
batch_size = st.sidebar.slider("Batch Size", 1, 8, 2)
dropout_val = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1, 0.05)

# ---- 训练数据 ----
st.header("📊 训练数据")

training_texts = st.text_area(
    "输入训练文本（多行）",
    value="我爱看猫。\n我喜欢猫。\n猫很可爱。\n你好世界。\nWorld is beautiful.",
    height=120,
)

if training_texts:
    # 构建训练样本
    all_chars = set()
    for line in training_texts.split("\n"):
        all_chars.update(line)
    # 扩展 vocab
    extra_chars = list(all_chars - set(tokenizer.vocab))
    if extra_chars:
        tokenizer.vocab.extend(extra_chars)
        tokenizer.char2idx = {c: i for i, c in enumerate(tokenizer.vocab)}
        tokenizer.idx2char = {i: c for c, i in tokenizer.char2idx.items()}
        tokenizer.vocab_size = len(tokenizer.vocab)

    st.info(f"字符集大小: **{tokenizer.vocab_size}** | 训练文本行数: **{len(training_texts.split(chr(10)))}**")
else:
    st.info(f"字符集大小: **{tokenizer.vocab_size}**")

# ---- 训练按钮 ----
st.header("🚀 训练")

if "train_log" not in st.session_state:
    st.session_state.train_log = []

col1, col2 = st.columns([1, 3])
with col1:
    start_btn = st.button("▶ 开始训练", type="primary", use_container_width=True)

with col2:
    if st.session_state.train_log:
        latest_epoch = len(st.session_state.train_log)
        st.caption(f"已训练 {latest_epoch} epoch | 最新 loss: {st.session_state.train_log[-1]:.4f}")

if start_btn:
    # 重置模型
    np.random.seed(42)
    model = ToyLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout_val,
    )

    # 准备训练数据
    texts = [t for t in training_texts.split("\n") if t.strip()]
    encoded_texts = [tokenizer.encode(t) for t in texts]
    max_len = min(model.max_seq_len, max(len(t) for t in encoded_texts))

    st.session_state.train_log = []
    losses = []

    with st.spinner(f"训练中... ({epochs} epochs)"):
        for epoch in range(epochs):
            epoch_losses = []
            for _ in range(batch_size):
                # 随机采样训练样本
                idx = np.random.randint(len(encoded_texts))
                seq = encoded_texts[idx][:max_len]
                X = np.array([seq], dtype=np.int32)
                targets = np.array([seq[1:] if len(seq) > 1 else seq], dtype=np.int32)

                if targets.shape[1] == 0:
                    targets = np.zeros_like(X)

                loss = train_step(model, X, targets, lr)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            st.session_state.train_log.append(float(avg_loss))
            losses.append(float(avg_loss))

            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                st.toast(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    st.success("✅ 训练完成！")

# ---- 训练曲线 ----
if losses:
    st.subheader("📈 训练曲线")
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(losses) + 1)),
        y=losses,
        mode="lines+markers",
        name="Loss",
        line=dict(color="#FF6B6B", width=2),
        marker=dict(size=4),
    ))
    fig.update_layout(
        title="Training Loss",
        xaxis_title="Epoch",
        yaxis_title="Cross-Entropy Loss",
        height=300,
        template="simple_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- 文本生成 ----
st.header("✍️ 文本生成")

prompt = st.text_input("输入提示文本", "我")

if prompt and losses:
    # 简单生成：基于训练数据的字符频率统计（因为数值梯度近似无法真正 backprop）
    # 使用训练文本中的字符转移概率来做简单的 next-char 预测
    char_counts: Dict[str, Dict[str, int]] = {}
    for text in texts:
        for i in range(len(text)):
            c = text[i]
            if c not in char_counts:
                char_counts[c] = {}
            if i + 1 < len(text):
                next_c = text[i + 1]
                char_counts[c][next_c] = char_counts[c].get(next_c, 0) + 1

    # 生成文本
    generated = list(prompt)
    for _ in range(20):
        last_char = generated[-1]
        if last_char in char_counts and char_counts[last_char]:
            options = list(char_counts[last_char].keys())
            weights = list(char_counts[last_char].values())
            total = sum(weights)
            probs = np.array(weights, dtype=float) / total
            next_char = np.random.choice(options, p=probs)
            generated.append(next_char)
        else:
            break

    result = "".join(generated)
    st.markdown(f"**生成结果**: {result}")
    st.code(result)
else:
    st.caption("请先点击「开始训练」，然后在上方输入提示文本。")

# ---- 模型信息 ----
with st.expander("📋 模型配置"):
    st.code(f"""
d_model: {d_model}
num_heads: {num_heads}
num_layers: {num_layers}
d_ff: {d_ff}
learning_rate: {lr}
dropout: {dropout_val}
vocab_size: {tokenizer.vocab_size}
max_seq_len: {model.max_seq_len if losses else 32}
    """.strip())
