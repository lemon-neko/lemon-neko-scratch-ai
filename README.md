# 羽白 AI Coding

> 从零实现与 AI 学习笔记的个人实验仓库（lemon-neko 的练手与笔记集）。

这是一个以"手写实现 + 交互式可视化 + 可运行示例"为核心的实验性仓库，目标是把每次学习、实现、复现与小实验都记录成可运行的 artefact，便于回顾、分享和教学。

---

## 🌟 亮点：可视化教学平台

### Streamlit Web 应用

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

### 静态 HTML 教学站

基于 Streamlit 页面重构的纯前端教学平台，支持独立部署（Vercel）：

```bash
cd ai-teaching-platform
# 直接打开 index.html 或在本地起一个静态服务器
```

---

## 核心模块一览

| 模块 | 说明 |
|------|------|
| **Self-Attention 详解** | 8 步交互式教程，对应 notebook 的完整流程 |
| **注意力热力图** | 可调 temperature/init 的注意力权重可视化 |
| **模型游乐场** | 配置超参数，训练玩具字符级语言模型 |
| **Transformer 块** | 完整 Encoder/Decoder Block 结构与张量形状 |
| **梯度流分析** | 手动反向传播与梯度范数可视化 |
| **项目概览** | 学习路线、核心公式、指标卡片 |

---

## 主要目标

- **手写实现（from-scratch）**：Self-Attention、Transformer、优化器等，深入理解模型工作原理
- **交互式可视化**：Streamlit Web 应用 + 静态 HTML 教学站，实时调节参数观察模型行为
- **可运行示例（notebooks/examples）**：包含 `.py` 脚本与交互式 `.ipynb`
- **单元测试**：pytest 覆盖核心模块，CI 自动验证

---

## 仓库结构

```
羽白-AI-Coding/
├── src/                  ← 从零实现的 ML 核心模块
│   ├── attention.py      ← Self-Attention（NumPy + 反向传播）
│   └── layers.py         ← LayerNorm, FFN, 位置编码
├── app/                  ← Streamlit 可视化教学应用
│   ├── main.py           ← 入口（首页 + 导航）
│   ├── pages/            ← 6 个教学页面
│   ├── components/       ← 可复用可视化组件
│   └── styles.py         ← CSS 设计令牌与主题
├── ai-teaching-platform/ ← 静态 HTML 教学站（Vercel 部署）
│   ├── index.html        ← 首页
│   ├── pages/            ← 各教学页面 HTML
│   ├── assets/           ← 图片资源
│   ├── colors_and_type.css ← 样式表
│   └── vercel.json       ← Vercel 部署配置
├── notebooks/            ← Jupyter 教程笔记本
├── tests/                ← pytest 单元测试
├── examples/             ← 独立可运行脚本（待补充）
├── data/                 ← 小型示例数据（待补充）
├── notes/                ← 学习笔记（待补充）
├── experiments/          ← 实验配置与结果（待补充）
├── tools/                ← 开发与部署脚本（待补充）
├── docs/                 ← 详细文档（待补充）
├── assets/               ← 通用静态资源（待补充）
├── requirements.txt      ← 运行时依赖
├── requirements-dev.txt  ← 开发依赖
└── .github/workflows/    ← CI
```

---

## 快速开始

### 1. 安装依赖

```bash
# 核心依赖
pip install -r requirements.txt          # numpy, streamlit, plotly
pip install -r requirements-dev.txt      # pytest, ruff（开发用）
```

### 2. 运行 Streamlit 应用

```bash
streamlit run app/main.py
```

### 3. 运行静态 HTML 教学站

```bash
# 直接浏览器打开
open ai-teaching-platform/index.html

# 或启动本地服务器
python -m http.server -d ai-teaching-platform 8000
```

### 4. 运行单元测试

```bash
pytest tests/ -v
```

### 5. 运行独立脚本

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
- 运行 `pytest` 单元测试（Python 3.10/3.11，仅 CPU）
- 使用 `ruff` 检查代码风格
- 运行 `src/` 模块的 `__main__` 测试

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **核心实现** | 纯 NumPy（手动反向传播） |
| **验证对照** | PyTorch（条件导入，仅用于对比） |
| **交互式可视化** | Streamlit + Plotly |
| **静态教学站** | 原生 HTML/CSS/JS + Vercel |
| **测试** | pytest |
| **代码风格** | ruff |
| **CI/CD** | GitHub Actions |

---

## 联系与声明

作者：lemon-neko

MIT 许可证。
