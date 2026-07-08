# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 提供本仓库的开发指导。

## 项目概述

这是一个个人 AI/ML 动手实验仓库，专注于使用 **纯 NumPy 从零实现 Transformer 组件**，并配套 **Streamlit 交互式可视化教学平台**。作者：lemon-neko，MIT 协议。

核心模块 (`src/`) 为纯 NumPy 实现，含手动反向传播；PyTorch 仅用于对比验证。Streamlit 应用 (`app/`) 提供 6 个教学页面：Self-Attention 详解、注意力热力图探索、玩具语言模型游乐场、Transformer 块结构可视化、梯度流分析等。

## 主要目录

| 目录 | 用途 |
|------|------|
| `src/` | 从零实现的 ML 核心模块（attention、layers） |
| `app/` | Streamlit Web 应用（main.py 入口、pages/、components/） |
| `tests/` | pytest 单元测试 |
| `notebooks/` | Jupyter 教程（主文件：`01-self-attention-from-scratch.ipynb`） |
| `examples/` | 独立可运行脚本 |
| `notes/` | 学习笔记 |
| `experiments/` | 实验配置（config.yaml、run.sh、results/） |
| `data/` | 小型示例数据 |
| `tools/` | 开发与部署脚本 |
| `docs/` | 详细文档 |

## 核心源码架构

**`src/attention.py`** — Self-Attention 实现（4 个版本）：
1. `SelfAttentionNumpy` — 单头纯 NumPy 实现，用于教学
2. 模块级函数：`split_heads`、`merge_heads`、`scaled_dot_product_attention`、`multi_head_attention_numpy`
3. `scaled_dot_product_attention_backward` — 手动反向传播梯度推导
4. `SelfAttentionFromScratch` — 完整 Transformer Block 类（MHA + 残差连接 + LayerNorm + FFN + Dropout）

**`src/layers.py`** — 基础神经网络构建块：
- `layer_norm` — Layer Normalization（支持可学习的 gamma/beta）
- `feed_forward` — 逐位置前馈网络（ReLU 激活，可选 dropout）
- `positional_encoding` — 正弦位置编码
- `causal_mask` / `padding_mask` — Decoder 因果掩码与 padding 掩码生成

## 开发命令

```bash
# 安装依赖
pip install -r requirements.txt          # 运行时：numpy, torch, streamlit, plotly
pip install -r requirements-dev.txt      # 开发：pytest, ruff, jupyter

# 运行测试
pytest tests/ -v

# Ruff 代码检查
ruff check src/ app/ tests/

# 启动 Streamlit 应用
streamlit run app/main.py

# 运行 src 模块（作为脚本）
python -m src.attention
python -m src.layers                     # 若存在 __main__

# 运行独立脚本
python notebooks/debug_self_attention.py
```

## CI（GitHub Actions）

配置文件：`.github/workflows/ci.yml`。在推送到 `main` 分支或创建 PR 时自动触发，使用 Ubuntu + Python 3.10/3.11：
- **test job**: `pytest tests/ -v --tb=short`（仅 CPU）
- **lint job**: `ruff check src/ app/ tests/`
- **examples job**: `python -m src.attention` + `import src.layers` 检查

## 编码规范

- `src/` 全部使用纯 NumPy；PyTorch 仅在 `if __name__ == "__main__"` 条件块中导入用于验证
- 大量使用带 LaTeX 数学公式（$$\dots$$）的 docstring
- 全程使用类型注解（Tuple、Optional 等）
- 模块顶部使用 `from __future__ import annotations`
- 类将权重存储为 `np.ndarray` 属性，使用 Xavier 缩放初始化
- `SelfAttentionFromScratch` 是最完整的类：支持训练模式的 dropout，维护 `W_Q/K/V/O` 和 FFN 的 `W_1/b_1/W_2/b_2`
- 测试使用 pytest 类（`TestClass`）加描述性方法名
- `app/main.py` 中有 `sys.path` 注入逻辑，兼容 Streamlit Cloud 部署
