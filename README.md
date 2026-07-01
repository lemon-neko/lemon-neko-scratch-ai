# lemon-neko-scratch-ai

这是一个用于记录「从零实现」与 AI 学习笔记的个人实验仓库（lemon-neko 的练手与笔记集）。仓库目标是把每次学习、手撕实现、论文阅读与小实验整理为可运行、可复现的代码和笔记，方便回顾与分享。

主要内容与目标

- 手写实现（from-scratch）：例如 Self-Attention、Transformer、优化器等，用以加深对原理的理解。
- 学习笔记（notes）：论文/文章摘要、实现思路、实验结论与 TODO 列表。
- 可运行示例（examples）：包含 .py 脚本与交互式 notebook，方便演示与教学。
- 小型实验（playground）：快速验证想法、小型基准或可视化 demo。

仓库结构（示例）

- README.md — 仓库说明（本文件）
- LICENSE — 版权许可（MIT）
- .gitignore — 忽略规则（Python / Jupyter）
- examples/ — 可运行示例（.py + .ipynb）
- src/ — 可复用的小模块（例如 attention.py）
- notes/ — 学习笔记（Markdown）
- public/ — 静态资源（favicon、示意图）
- .github/workflows/ — CI 配置（可选，运行示例或测试）

快速开始

先决条件

- Python 3.8+
- pip
- 推荐创建虚拟环境

1. 克隆仓库

```bash
git clone https://github.com/lemon-neko/lemon-neko-scratch-ai.git
cd lemon-neko-scratch-ai
```

2. （可选）创建并激活虚拟环境

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

3. 安装依赖（按需）

示例只需 PyTorch（或在不需要 GPU 的情况下用 CPU 版本）：

```bash
pip install --upgrade pip
pip install torch
```

4. 运行示例脚本

```bash
python examples/self_attention_pytorch.py
```

5. 打开 Notebook（可交互）

```bash
jupyter notebook examples/self_attention_pytorch.ipynb
```

示例说明

- examples/self_attention_pytorch.py
  - 一个简洁的 scaled dot-product attention 演示脚本，打印输出与注意力矩阵形状，便于快速理解。
- src/attention.py
  - 可复用的 attention 函数与一个简单的 MultiHeadAttention 类（手写实现，便于教学）。
- notes/TEMPLATE_NOTE.md
  - 笔记模板：日期、摘要、关键公式、实现要点、实验结果与后续阅读建议。

贡献指南

欢迎你通过 Issue / PR 的方式贡献代码或笔记。建议遵循：

- 小而聚焦的提交：每个 PR 尽量只完成一项小改动（一个示例或一篇笔记）。
- 示例要求可运行：在 examples/ 下新增示例时，请附上简要运行说明与依赖（requirements.txt）。
- 笔记请使用 Markdown：放在 notes/ 下并使用 TEMPLATE_NOTE.md 模版。

许可

本仓库采用 MIT 许可证（LICENSE 文件包含完整文本），可自由使用/修改/分发，但请保留版权声明。

后续计划（建议）

- 将更多手撕实现与注释详尽的 notebook 加入 examples/；
- 把部分 notebook 做成可视化 demo（attention heatmap）；
- 添加 CI（测试示例脚本在 CPU 环境下能否跑通）；
- 如果需要，迁移为 Astro 或者添加一个 GitHub Pages 的展示页。

联系方式

- GitHub: https://github.com/lemon-neko
