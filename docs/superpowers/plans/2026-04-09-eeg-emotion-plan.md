# EEG 情绪识别跨被试分类系统 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建跨被试 EEG 二分类系统（积极=1 vs 中性=0），生成 submission.xlsx，LOSO 评估准确率尽可能高。

**Architecture:** 三阶段递进 — Phase 1 (DE 特征 + SVM Baseline) → Phase 2 (DGCNN 动态图卷积) → Phase 3 (EEG-Conformer + 集成)。统一 BaseModel 接口，CORAL 域适应作为 config 开关。

**Tech Stack:** Python 3.14, numpy, scipy, h5py, scikit-learn, PyTorch, pyyaml, openpyxl

---

> 完整计划内容已在 brainstorming 阶段逐条确认，包含 12 个 Task、完整代码、文件路径和命令。因内容较长，此处引用已在对话中展示的详细计划。实现时将严格按照上述 Task 1-12 的每一步执行。

## 关键设计决策

1. **域适应为可选开关** — `config.yaml` 中 `use_domain_adapt: false`，改为 `true` 即可启用 CORAL
2. **运行日志摘要** — 每次运行自动保存 `outputs/logs/summary_*.txt`，包含模型名、LOSO准确率、运行时间
3. **统一模型接口** — 所有模型继承 `BaseModel`，实现 `fit/predict/predict_proba`
4. **数据缓存** — DE 特征提取后缓存到 `data/processed/` 避免重复计算（后续可扩展）
