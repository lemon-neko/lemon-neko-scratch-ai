# lemon-neko-scratch-ai

> 从零实现与 AI 学习笔记的个人实验仓库（lemon-neko 的练手与笔记集）。

这是一个以“手写实现 + 阅读笔记 + 可运行示例”为核心的实验性仓库，目标是把每次学习、实现、复现与小实验都记录成可运行的 artefact，便于回顾、分享和教学。

---

## 主要目标

- 手写实现（from-scratch）：例如 Self-Attention、Transformer、优化器等，用以深入理解模型工作原理。
- 学习笔记（notes）：论文/文章摘录、读书笔记、实现要点与调参心得。
- 可运行示例（examples）：包含 `.py` 脚本与交互式 `.ipynb`，既可在本地跑也可用于教学演示。
- 实验记录（experiments）：保存实验配置、运行脚本与结果摘要，保证可复现性。

---

## 当前仓库结构（已模块化）

- `README.md` — 仓库总览（本文件）
- `LICENSE` — MIT 许可
- `.gitignore` — 常用忽略规则（Python / Jupyter）
- `src/` — 可复用模块（手写实现放这里，例如 `attention.py`）
- `examples/` — 可运行示例（脚本和 notebook）
- `notebooks/` — 交互式 notebook 集中区（教学 / 演示）
- `notes/` — 学习笔记、论文摘要（建议以 `YYYY-MM-DD-title.md` 命名）
- `experiments/` — 实验配置、结果与日志（按实验名分目录）
- `data/` — 小型示例数据或下载脚本（不建议提交大型数据）
- `assets/` — 静态资源（图、favicon、og 图等）
- `tools/` — 开发与部署脚本（如环境准备、测试脚本）
- `.github/workflows/` — CI（示例已包含运行 examples 的 workflow）

上述子目录均包含 README 占位说明，后续请按模块约定提交代码与笔记。

---

## 快速开始

1. 克隆仓库并进入目录：

```bash
git clone https://github.com/lemon-neko/lemon-neko-scratch-ai.git
cd lemon-neko-scratch-ai
```

2. 建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell
```

3. 安装示例依赖（按需）

```bash
pip install --upgrade pip
pip install torch
```

4. 运行示例脚本：

```bash
python examples/self_attention_pytorch.py
```

5. 打开 Notebook（可交互演示）：

```bash
jupyter notebook notebooks/  # 或直接打开 examples 下的 .ipynb
```

---

## 贡献说明

- 新示例放 `examples/`，同时提供 `.py`（用于自动化运行/CI）与 `.ipynb`（教学演示）优先级更高。
- 可复用组件放 `src/`，避免在示例中复制代码。
- 笔记放 `notes/`，建议使用 `TEMPLATE_NOTE.md` 作为参考格式。
- 实验放 `experiments/<name>/`，至少包含 `config.yaml`、`run.sh` 与 `results/README.md`。
- 提交前请确保示例脚本在 CPU 环境下可运行（或在 README 中标注特殊依赖）。

---

## 开发与 CI

仓库已包含一个示例 GitHub Actions workflow（`.github/workflows/ci.yml`），用于在推送/PR 时尝试运行示例脚本。你可以：

- 修改 workflow 中的依赖版本以保证 CI 在 CPU 环境下稳定运行（例如锁定 `torch` 版本）。
- 将更多测试用例加入 CI（例如对 `src` 模块的单元测试）。

---

## 计划与待办（示例）

- [ ] 丰富 `examples/self_attention_pytorch.ipynb`，加入 attention heatmap 可视化。
- [ ] 把更多 hand-written 的模块抽到 `src/`，并写好单元测试。
- [ ] 在 `docs/` 或 GitHub Pages 上做一个简易展示页。

---

## 联系与声明

作者：lemon-neko

如需引用或复现，请保留本仓库的版权信息（MIT 许可证）。
