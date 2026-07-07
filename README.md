# lemon-neko-scratch-ai

> 从零实现与 AI 学习笔记的个人实验仓库（lemon-neko 的练手与笔记集）。

这是一个以"手写实现 + 交互式可视化 + 可运行示例"为核心的实验性仓库，目标是把每次学习、实现、复现与小实验都记录成可运行的 artefact，便于回顾、分享和教学。

---

## 🌟 亮点：可视化教学平台

本项目包含一个 **Streamlit 交互式 Web 应用**，将 Transformer 原理以可视化方式呈现：

```bash
pip install -r requirements-app.txt
streamlit run app/main.py
```

**包含页面：**

| 页面 | 说明 |
|------|------|
| **首页** | 项目概览、学习路线、核心公式 |
| **Self-Attention 详解** | 8 步交互式教程，对应 notebook 的完整流程 |
| **注意力探索器** | 可调 temperature/init 的注意力权重可视化 |
| **模型游乐场** | 配置超参数，训练玩具字符级语言模型 |
| **Transformer 块** | 完整 Encoder/Decoder Block 结构与张量形状 |
| **梯度流分析** | 手动反向传播与梯度范数可视化 |

---

## 主要目标

- **手写实现（from-scratch）**：Self-Attention、Transformer、优化器等，深入理解模型工作原理
- **交互式可视化**：Streamlit Web 应用，实时调节参数观察模型行为
- **学习笔记（notes）**：论文/文章摘录、读书笔记、实现要点与调参心得
- **可运行示例（examples）**：包含 `.py` 脚本与交互式 `.ipynb`
- **实验记录（experiments）**：保存实验配置、运行脚本与结果摘要

---

## 仓库结构

```
lemon-neko-scratch-ai/
├── src/                  ← 从零实现的 ML 核心模块
│   ├── attention.py      ← Self-Attention（NumPy + 反向传播）
│   └── layers.py         ← LayerNorm, FFN, 位置编码
├── app/                  ← Streamlit 可视化教学应用
│   ├── main.py           ← 入口
│   ├── pages/            ← 各教学页面（6 个）
│   └── components/       ← 可复用可视化组件
├── notebooks/            ← Jupyter 教程笔记本
├── examples/             ← 独立可运行脚本
├── tests/                ← 单元测试
├── notes/                ← 学习笔记
├── experiments/          ← 实验配置与结果
├── data/                 ← 小型示例数据
├── assets/               ← 静态资源
├── tools/                ← 开发与部署脚本
├── docs/                 ← 详细文档
├── requirements-app.txt  ← 依赖清单
└── .github/workflows/    ← CI
```

---

## 快速开始

### 1. 安装依赖

```bash
# 核心 ML 模块
pip install numpy torch matplotlib

# Web 可视化应用
pip install -r requirements-app.txt
```

### 2. 运行 Streamlit 应用

```bash
streamlit run app/main.py
```

### 3. 运行单元测试

```bash
pytest tests/ -v
```

### 4. 运行独立脚本

```bash
python -m src.attention
python notebooks/debug_self_attention.py
```

---

## 贡献说明

- 新示例放 `examples/`，同时提供 `.py`（用于自动化运行/CI）与 `.ipynb`（教学演示）
- 可复用组件放 `src/`，避免在示例中复制代码
- 笔记放 `notes/`
- 实验放 `experiments/<name>/`，至少包含 `config.yaml`、`run.sh` 与 `results/README.md`
- 提交前请确保示例脚本在 CPU 环境下可运行

---

## 开发与 CI

仓库已包含 GitHub Actions workflow（`.github/workflows/ci.yml`），在推送/PR 时自动：
- 运行 `pytest` 单元测试
- 使用 `ruff` 检查代码风格
- 运行 `src/` 模块的 `__main__` 测试

---

## 联系与声明

作者：lemon-neko

如需引用或复现，请保留本仓库的版权信息（MIT 许可证）。
