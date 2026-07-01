experiments/

用于放置实验相关的配置、日志和结果（scripts、trials、checkpoint info）。

说明：
- experiments/<experiment-name>/ 下建议包含 run.sh、config.yaml、results/ 等子结构。
- 尽量不要把大型数据或模型权重直接提交到仓库，使用 data/ 或外部存储。
