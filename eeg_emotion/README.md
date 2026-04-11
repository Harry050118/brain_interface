# EEG Emotion Recognition

跨被试脑电情绪识别系统 — 脑机接口竞赛赛题四

> **任务**: 基于 30 通道 EEG 信号，识别被试的积极情绪（1）与中性情绪（0）
> **方法**: 留一被试交叉验证（LOSO） + 模型集成

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Input: EEG Data                              │
│              训练集: 60 被试 × 50 秒 / 被试 (30 通道, 250 Hz)        │
│              测试集: 10 被试 × 8 trials × 10 秒 / trial              │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
         ┌──────────────────┐        ┌──────────────────────┐
         │  Phase 1: SVM    │        │  Phase 3: Conformer   │
         │  DE 特征 (120维)  │        │  原始 EEG (30×2500)   │
         │  RBF Kernel      │        │  SpatialConv + MHA    │
         └────────┬─────────┘        └─────────┬────────────┘
                  │                            │
                  ▼                            │
         ┌──────────────────┐                   │
         │  Phase 2: DGCNN  │                   │
         │  DE 特征 → 动态图  │                   │
         └────────┬─────────┘                   │
                  │                            │
                  └─────────────┬──────────────┘
                                ▼
                    ┌───────────────────────┐
                    │  Ensemble: 加权投票    │
                    │  SVM:0.3 DGCNN:0.35   │
                    │  Conformer:0.35       │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │  Output: submission   │
                    │  80 predictions       │
                    └───────────────────────┘
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# Phase 1: SVM Baseline (最快验证)
python run_baseline.py

# Phase 2: 深度学习模型
python run_deep.py --model all

# 全链路: Baseline + Deep + Ensemble (推荐)
python run_all.py
```

## 实际运行结果

### LOSO 交叉验证准确率（10 折分层抽样: 7 HC + 3 DEP）

| 模型 | 输入 | 平均准确率 | 标准差 | 训练时间 |
|------|------|-----------|--------|---------|
| SVM + DE | 差分熵特征 (120 维) | 65.83% | ±13.5% | ~7 秒 |
| DGCNN | DE 特征 → 动态图 (30×4) | 60.83% | ±12.9% | ~5 分钟 |
| EEG-Conformer | 原始 EEG (30×2500) | 57.92% | ±8.2% | ~58 分钟 |

### 各被试详细结果

**SVM Baseline (65.83%)**

| Subject | Accuracy | Subject | Accuracy |
|---------|----------|---------|----------|
| DEP1003 | 68.06% | HC1003 | 80.56% |
| DEP1030 | 50.00% | HC1011 | 72.22% |
| DEP1032 | 54.17% | HC1014 | 50.00% |
| HC1017 | 95.83% | HC1054 | 62.50% |
| HC1025 | 62.50% | HC1035 | 62.50% |

**DGCNN (60.83%)**

| Subject | Accuracy | Subject | Accuracy |
|---------|----------|---------|----------|
| DEP1003 | 51.39% | HC1003 | 69.44% |
| DEP1030 | 50.00% | HC1011 | 61.11% |
| DEP1032 | 56.94% | HC1014 | 51.39% |
| HC1017 | 79.17% | HC1054 | 40.28% |
| HC1025 | 81.94% | HC1035 | 63.89% |

**EEG-Conformer (57.92%)**

| Subject | Accuracy | Subject | Accuracy |
|---------|----------|---------|----------|
| DEP1003 | 59.72% | HC1003 | 51.39% |
| DEP1030 | 58.33% | HC1011 | 56.94% |
| DEP1032 | 50.00% | HC1014 | 55.56% |
| HC1017 | 73.61% | HC1054 | 48.61% |
| HC1025 | 52.78% | HC1035 | 72.22% |

> 注：深度学习模型在小样本（每个被试约 72 个窗口）上训练 100 轮，存在一定过拟合。增加数据量或正则化可进一步提升。

---

## 项目结构

```
eeg_emotion/
├── configs/
│   └── config.yaml          # 所有超参数（数据路径、模型参数等）
├── src/                     # 核心模块
│   ├── data_loader.py       # 训练集(h5py v7.3) + 测试集(scipy v7) 读取
│   ├── features.py          # 带通滤波 + 差分熵(DE)特征提取
│   ├── domain_adapt.py      # CORAL 域适应（协方差对齐）
│   ├── train.py             # 分层 LOSO 交叉验证
│   ├── predict.py           # 测试集预测 + 集成推理
│   ├── ensemble.py          # 模型集成（加权投票 / 多数投票）
│   └── models/
│       ├── base_model.py    # 抽象基类（fit/predict/predict_proba）
│       ├── svm_model.py     # SVM (RBF Kernel + StandardScaler)
│       ├── dgcnn.py         # 动态图卷积神经网络 (Attention-based Adjacency)
│       └── eeg_conformer.py # 空间卷积 + Transformer (端到端)
├── run_baseline.py          # 一键运行 SVM Baseline
├── run_deep.py              # 一键运行深度学习模型
├── run_all.py               # 全链路运行
├── save_models.py           # 训练并保存模型权重
├── requirements.txt         # Python 依赖
├── README.md                # 本文件
└── outputs/
    ├── models/              # 训练好的模型权重
    │   ├── svm.pkl          # SVM 模型 (pickle)
    │   ├── dgcnn.pt         # DGCNN 权重 (PyTorch)
    │   └── conformer.pt     # EEG-Conformer 权重 (PyTorch)
    ├── logs/                # 运行日志
    └── submission.xlsx      # 测试集预测结果
```

---

## 模型详解

### 1. SVM + DE 特征

**原理**: 使用差分熵（Differential Entropy）作为特征表示。
- 对每个通道，分别计算 4 个频段的 DE：theta(4-8Hz)、alpha(8-13Hz)、beta(13-30Hz)、gamma(31-49Hz)
- 30 通道 × 4 频段 = **120 维特征向量**
- 特征经 StandardScaler 标准化后输入 RBF 核 SVM

**架构**:
```
EEG (30 ch × 2500 timesteps)
    │
    ├── Bandpass Filter (Butterworth, 4 bands)
    │
    └── Differential Entropy → 120-dim vector
            │
            └── StandardScaler → SVM (RBF, C=1.0) → {0, 1}
```

### 2. DGCNN (Dynamic Graph Convolutional Neural Network)

**原理**: 将 30 个 EEG 通道视为图的 30 个节点，通过学习动态邻接矩阵来捕获通道间的功能连接。
- 每个节点的初始特征为该通道在 4 个频段上的 DE 值
- 通过注意力机制动态学习通道间的连接权重
- 多层图卷积捕获高阶空间关系

**架构**:
```
DE Features (30 × 4)
    │
    ├── Input Projection → (30 × hidden_dim)
    │
    ├── DynamicGraphConv (Attention-based Adjacency) × 3
    │   └── A_ij = softmax(LeakyReLU(a^T [h_i || h_j]))
    │
    └── FC(30×hidden → 128 → 2) → {0, 1}
```

### 3. EEG-Conformer

**原理**: 结合 CNN（空间特征提取）与 Transformer（时间建模），直接在原始 EEG 上进行端到端训练。
- 空间卷积使用 (30, 1) 的 2D 卷积核，一次性提取跨通道的空间模式
- 时间池化将 2500 个时间步降至 250，避免自注意力的 O(n²) 显存开销
- 4 层 Conformer Block (MHSA + FFN) 捕获时序依赖

**架构**:
```
Raw EEG (batch × 30 ch × 2500 timesteps)
    │
    ├── SpatialConv (Conv2d 1×30×1) → (batch × hidden × 2500)
    │
    ├── Time Pooling (mean, pool=10) → (batch × 250 × hidden)
    │
    ├── Linear Projection → (batch × 250 × hidden)
    │
    ├── ConformerBlock (MHA + FFN) × 4
    │
    └── Global Avg Pool → FC(hidden→32→2) → {0, 1}
```

### 4. 模型集成

采用**加权投票**策略，综合三个模型的预测概率：
```
P_final = 0.30 × P_SVM + 0.35 × P_DGCNN + 0.35 × P_Conformer
label = argmax(P_final)
```

---

## 配置参数

所有参数统一在 `configs/config.yaml` 中管理：

| 类别 | 关键参数 | 默认值 | 说明 |
|------|---------|--------|------|
| 数据 | train_dir | ../训练集 | 训练数据路径 |
| 数据 | test_dir | ../公开测试集 | 测试数据路径 |
| 信号 | sample_rate | 250 | 采样率 (Hz) |
| 信号 | n_channels | 30 | 通道数 |
| 信号 | window_size_sec | 10 | 窗口长度 (秒) |
| 训练 | n_eval_subjects | 10 | LOSO 评估被试数 |
| SVM | C | 1.0 | SVM 正则化强度 |
| DGCNN | hidden_dim | 64 | 隐藏层维度 |
| DGCNN | num_layers | 3 | 图卷积层数 |
| Conformer | hidden_dim | 64 | Transformer 维度 |
| Conformer | num_heads | 4 | 注意力头数 |
| Conformer | num_layers | 4 | Transformer 层数 |
| Conformer | time_pool | 10 | 时间下采样倍数 |
| 域适应 | use_domain_adapt | false | CORAL 开关 |

---

## 域适应 (CORAL)

**CORAL (CORrelation ALignment)**: 通过匹配源域和目标域的二阶统计量（协方差矩阵），将源域特征对齐到目标域分布。

**启用方法**:
```yaml
# configs/config.yaml
domain_adapt:
  use_domain_adapt: true  # 将 false 改为 true
```

**原理**:
```
源域 (训练被试): X_source ~ (μ_s, Σ_s)
目标域 (测试被试): X_target ~ (μ_t, Σ_t)

对齐变换: X_aligned = X_source @ Σ_s^{-1/2} @ Σ_t^{1/2}
```

无需修改任何训练代码，仅需切换配置开关。

### CORAL 实验结果

对 CORAL 域适应进行了完整对比测试：

| 模型 | 不使用 CORAL | 使用 CORAL | 差异 |
|------|------------|-----------|------|
| SVM | **65.83%** | 50.00% | **-15.83%** |
| DGCNN | **60.83%** | 49.31% | **-11.53%** |

**结论**: CORAL 在当前数据集上**显著降低了性能**，所有被试准确率均降至随机猜测水平（~50%）。

**原因分析**: CORAL 需要可靠地估计源域和目标域的协方差矩阵。在当前 LOSO 设置下，每个被试仅约 72 个窗口，样本量太小导致协方差估计不稳定，对齐反而引入了噪声。

**建议**: 保持 `use_domain_adapt: false`。当未来获得更大规模的测试数据时，可重新评估 CORAL 的效果。

---

## 模型加载与使用

```python
import sys
sys.path.insert(0, "src/models")

# 加载模型
from models.svm_model import SVMModel
from models.dgcnn import DGCNNModel
from models.eeg_conformer import EEGConformerModel

svm = SVMModel.load("outputs/models/svm.pkl")
dgcnn = DGCNNModel.load("outputs/models/dgcnn.pt")
conformer = EEGConformerModel.load("outputs/models/conformer.pt")

# 推理
import numpy as np
X_de = np.random.randn(1, 120)  # 120 维 DE 特征
X_raw = np.random.randn(1, 30, 2500)  # 原始 EEG

print("SVM prediction:", svm.predict(X_de))
print("DGCNN prediction:", dgcnn.predict(X_de))
print("Conformer prediction:", conformer.predict(X_raw))
```

---

## 依赖

- numpy, scipy, h5py — 数据处理
- scikit-learn — SVM + 评估
- pytorch — DGCNN + EEG-Conformer
- pyyaml — 配置管理
- openpyxl — 生成 submission.xlsx

> 建议使用 conda 环境（Python 3.10），以确保 CUDA PyTorch 兼容性。

---

## 环境

- **Python**: 3.10 (conda env: llm)
- **GPU**: NVIDIA RTX 5060 (8GB)
- **PyTorch**: CUDA-enabled
- **OS**: Windows 11

---

## 完整训练流程

```bash
# 一键运行: SVM + DGCNN + Conformer + Ensemble
cd "D:/college/智医工综合实验二/project/eeg_emotion"
python run_all.py

# 仅训练并保存模型（不运行 LOSO 评估）
python save_models.py
```

---

## 输出文件

| 文件 | 内容 |
|------|------|
| `outputs/models/svm.pkl` | SVM 模型权重 + StandardScaler |
| `outputs/models/dgcnn.pt` | DGCNN 权重 + 归一化参数 |
| `outputs/models/conformer.pt` | Conformer 权重 + 归一化参数 |
| `outputs/submission.xlsx` | 测试集预测结果 (80 条) |
| `outputs/logs/run_*.log` | 每次运行的完整日志 |
| `outputs/logs/summary_*.txt` | 运行结果摘要 |
| `outputs/plots/training_curves.png` | 三个模型的训练曲线对比 |
| `outputs/plots/training_*.png` | 各模型单独的训练曲线 |

## 辅助脚本

| 脚本 | 功能 |
|------|------|
| `run_all.py` | 全链路：SVM → DGCNN → Conformer → Ensemble |
| `save_models.py` | 仅训练并保存模型（不运行 LOSO） |
| `plot_curves.py` | 训练模型并生成准确率/损失曲线图 |
| `test_coral.py` | CORAL 域适应对比测试 |
