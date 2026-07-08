"""
Streamlit 主入口
=================

启动方式：
    streamlit run app/main.py

页面导航通过 app/pages/ 目录下的 Python 文件自动注册。
"""

import os
import sys

# 确保 src/ 和 app/ 下的模块可以被导入
# 本地开发时 sys.path 已有仓库根，Cloud 上需要显式添加
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from styles import STYLESHEET

import streamlit as st

st.set_page_config(
    page_title="AI 全栈可视化教学平台",
    page_icon="🐱",
    layout="wide",
)

st.title("🐱 AI 全栈可视化教学平台")

st.markdown(STYLESHEET, unsafe_allow_html=True)

st.markdown("""
从零实现 Transformer 组件，并搭配交互式可视化，帮助你深入理解 AI 的内部原理。

### 快速开始

| 页面 | 说明 |
|------|------|
| **① 首页** | 项目概览、目录结构、学习路线 |
| **② Self-Attention 详解** | 交互式逐步讲解注意力机制 |
| **③ 注意力热力图** | 可视化注意力权重分布 |
| **④ 模型游乐场** | 配置超参数，训练玩具语言模型 |
| **⑤ Transformer 块** | 编码器/解码器块的结构可视化 |
| **⑥ 梯度流分析** | 手动反向传播与梯度范数展示 |

### 运行本地

```bash
pip install -r requirements-app.txt
streamlit run app/main.py
```

### 项目结构

```
lemon-neko-scratch-ai/
├── src/                  ← 从零实现的 ML 模块
│   ├── attention.py      ← Self-Attention 核心
│   └── layers.py         ← LayerNorm, FFN, 位置编码
├── app/                  ← Streamlit 可视化教学应用
│   ├── main.py           ← 入口
│   ├── pages/            ← 各教学页面
│   └── components/       ← 可复用可视化组件
├── notebooks/            ← Jupyter 教程笔记本
├── examples/             ← 独立可运行脚本
└── tests/                ← 单元测试
```

---
*基于 "Attention Is All You Need" 论文，纯 NumPy 实现 + PyTorch 验证。*
""")
