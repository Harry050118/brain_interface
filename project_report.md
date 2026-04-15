# 基于 EEG-Conformer 与类别先验后处理的跨被试脑电情绪识别方法研究

## 摘要

本文面向“基于脑电数据的情绪识别算法”赛题，研究跨被试 EEG 情绪二分类任务，即根据 30 通道脑电信号识别中性情绪与积极情绪。该任务的主要挑战在于 EEG 信号噪声较高、被试间差异显著、训练样本规模有限。本文采用 BD-Conformer 对原始 EEG 窗口进行端到端建模，并结合窗口级标准化和基于公开实验设计的 balanced-rank 后处理。full LOSO 验证结果显示，窗口级 balanced-rank 达到 73.80%；进一步在训练集 trial-level 评估中聚合同一 trial 的多个窗口概率后，准确率达到 84.17%。由于公开测试集真实标签不公开，本文所有监督指标均来自训练集 full LOSO，公开测试集仅用于无标签推理和 submission 生成。

## 1. 引言

脑电情绪识别通过分析 EEG 信号判断被试情绪状态。相比表情、语音等外显信号，EEG 更接近神经活动本身，但也具有低信噪比、强个体差异和小样本等特点，跨被试识别尤其困难。

传统方法通常提取 DE 等手工特征并使用 SVM 分类，解释性较强，但难以充分建模 EEG 的空间-时间关系。本文最终采用 EEG-Conformer 类模型，直接输入原始 EEG 窗口，通过卷积和自注意力提取时空特征。同时，赛题数据说明公开了每名被试包含 4 段中性视频和 4 段积极视频，因此本文加入 balanced-rank 后处理，使同一被试的预测类别比例符合公开实验设计。

本文主要工作包括：

1. 使用 BD-Conformer 对原始 30 通道 EEG 进行跨被试情绪识别。
2. 使用 window normalization 缓解被试间幅值差异。
3. 使用 balanced-rank 和 trial-level balanced-rank 后处理提高预测稳定性。
4. 通过 full LOSO 验证模型，并明确区分窗口级结果与 trial-level 聚合结果。

## 2. 数据集与任务定义

训练集包含 60 名被试，其中健康被试 40 名、抑郁症被试 20 名。每名被试观看 8 段视频，包括 4 段中性视频和 4 段积极视频。训练文件中 `EEG_data_neu` 对应标签 0，`EEG_data_pos` 对应标签 1；每类数据为 `30 x 50000`，即 4 段约 50 秒 EEG。采样率为 250 Hz。

公开测试集包含 10 名被试，每名被试同样包含 8 个 trial，其中 4 个中性、4 个积极，但真实标签不公开。每个测试文件为 `30 x 20000`，即 8 个 10 秒 trial。提交文件需包含 `user_id`、`trial_id` 和 `Emotion_label`。

| 数据集 | 被试数 | 每名被试 trial 数 | 类别比例 | 单 trial 长度 | 标签 |
|---|---:|---:|---:|---:|---|
| 训练集 | 60 | 8 | 4 中性 / 4 积极 | 约 50s | 公开 |
| 公开测试集 | 10 | 8 | 4 中性 / 4 积极 | 10s | 隐藏 |

## 3. 方法

### 3.1 方法流程

整体流程如图 1 所示。原始 EEG 首先被切分为 10 秒窗口并进行窗口级标准化；随后送入 BD-Conformer 输出类别概率；最后根据每名被试 4 个中性 trial 和 4 个积极 trial 的公开先验进行 balanced-rank 后处理，生成最终预测标签。

![图 1 方法整体流程](eeg_emotion/outputs/figures/method_pipeline.png)

主要功能包括：

1. **信号预处理**：将原始 EEG 转换为固定长度窗口，并进行通道级标准化。
2. **时空特征学习**：使用 BD-Conformer 学习 EEG 空间-时间特征。
3. **先验约束后处理**：使用 balanced-rank 保证同一被试预测结果符合 4/4 类别比例。

主要创新点包括：

1. 将 EEG-Conformer 与窗口级标准化结合，用于小样本跨被试 EEG 分类。
2. 基于赛题公开类别比例设计 balanced-rank 后处理。
3. 在训练集验证中引入 trial-level 概率聚合，降低单窗口预测噪声。

### 3.2 数据预处理

训练集中每个 50 秒 trial 被切分为 10 秒窗口，窗口长度为 2500 个采样点，stride 为 1250 个采样点。因此，每个训练 trial 可得到 9 个重叠窗口，窗口输入维度为 `30 x 2500`。

为降低不同被试之间 EEG 幅值差异，本文采用 window normalization：对每个窗口内每个通道沿时间维分别计算均值和标准差，再进行标准化。

```python
def standardize_by_window(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = X.mean(axis=-1, keepdims=True).astype(np.float32)
    std = X.std(axis=-1, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return ((X - mean) / std).astype(np.float32, copy=False)
```

### 3.3 BD-Conformer 模型

本文使用 Braindecode 中的 EEGConformer。该模型结合卷积模块与自注意力模块：卷积用于提取局部时空模式，自注意力用于建模较长程依赖。模型输入为原始 EEG 窗口，输出为两类 logits。

| 项目 | 设置 |
|---|---|
| 输入维度 | `30 x 2500` |
| 输出类别 | 2 |
| Dropout | 0.5 |
| 优化器 | AdamW |
| 损失函数 | Cross entropy |

### 3.4 Balanced-rank 后处理

数据说明文档明确每名被试包含 4 个中性 trial 和 4 个积极 trial。本文利用这一公开先验，对同一被试的 8 个 trial 按积极类概率排序，选择概率最高的 4 个作为积极类，其余作为中性类。该步骤不使用测试集真实标签。

```python
def balanced_rank_predictions(probas):
    probas = np.asarray(probas)
    n_positive = probas.shape[0] // 2
    preds = np.zeros(probas.shape[0], dtype=np.int64)
    positive_rank = np.argsort(probas[:, 1], kind="mergesort")
    preds[positive_rank[-n_positive:]] = 1
    return preds
```

### 3.5 Trial-level balanced-rank

训练集中的一个 trial 约 50 秒，可切分为 9 个 10 秒窗口。trial-level balanced-rank 先对同一 trial 的窗口概率取平均，再对 8 个 trial 做 balanced-rank。该策略用于训练集 LOSO 的 trial-level 验证，能够减少单窗口波动。

```python
def trial_balanced_rank_predictions(probas, windows_per_trial):
    trial_probas = probas.reshape(-1, windows_per_trial, probas.shape[1]).mean(axis=1)
    trial_preds = balanced_rank_predictions(trial_probas)
    return np.repeat(trial_preds, windows_per_trial)
```

需要注意，公开测试集中每个 trial 仅提供 10 秒 EEG，不能像训练集一样进行 9 窗口聚合。因此 trial-level 结果更适合作为训练集 trial 结构下的验证指标，而不是公开测试集准确率。

## 4. 实验设置

本文采用 full LOSO 交叉验证。每一折留出 1 名训练被试作为验证集，其余 59 名被试用于训练，遍历全部 60 名训练被试。

| 参数 | 设置 |
|---|---|
| 模型 | BD-Conformer |
| 输入 | Raw EEG, `30 x 2500` |
| 窗口长度 | 10s |
| 训练 stride | 5s |
| 标准化 | Window normalization |
| Dropout | 0.5 |
| Epochs | 20 |
| Early stopping patience | 3 |
| 后处理 | balanced-rank / trial-balanced-rank |
| 评估方式 | Full LOSO |

## 5. 实验结果与可视化

本节中的准确率、逐被试柱状图、准确率分布图和混淆矩阵均来自训练集 full LOSO 交叉验证。公开测试集真实标签不公开，因此不能在本地计算公开测试集准确率或混淆矩阵；公开测试集仅用于无标签推理和 submission 生成。

### 5.1 方法准确率对比

图 2 和表 1 展示了主要方法结果。窗口级 balanced-rank full LOSO 达到 73.80%；trial-level balanced-rank 在训练集 trial 聚合口径下达到 84.17%。

![图 2 方法准确率对比](eeg_emotion/outputs/figures/method_accuracy_comparison.png)

| 方法 | 评估口径 | Full LOSO Accuracy | 说明 |
|---|---|---:|---|
| SVM/早期基线 | window-level | 65.83% | 传统机器学习基线 |
| BD-Conformer + window norm | window-level | 71.71% | 原始 EEG 深度模型 |
| BD-Conformer + window norm + balanced-rank | window-level | 73.80% | 保守窗口级结果 |
| BD-Conformer + window norm + trial-balanced-rank | trial-level | 84.17% | 训练集 trial 聚合评估 |

### 5.2 被试级 LOSO 结果

图 3 展示 60 名训练被试的 trial-level LOSO 准确率。该图由 full LOSO 日志中的每折 `best_acc` 解析得到。可以看到，模型在部分被试上达到 100%，但也有少数被试低于 50%，说明跨被试个体差异仍然明显。

![图 3 逐被试 full LOSO 准确率](eeg_emotion/outputs/figures/subject_loso_accuracy.png)

### 5.3 准确率分布

图 4 展示 trial-level LOSO 准确率分布。该实验平均准确率为 84.17%，标准差为 16.44%。标准差较大说明模型仍存在被试依赖性，后续需要更稳健的跨被试泛化方法。

![图 4 被试准确率分布](eeg_emotion/outputs/figures/accuracy_distribution.png)

### 5.4 Trial-level LOSO 混淆矩阵

图 5 为训练集 trial-level full LOSO 验证集上的混淆矩阵，不是公开测试集混淆矩阵。由于每名验证被试真实包含 4 个中性 trial 和 4 个积极 trial，且后处理同样预测 4 个中性和 4 个积极，因此可由逐被试 trial-level accuracy 汇总得到整体 TP、TN、FP 和 FN。

![图 5 Trial-level LOSO 混淆矩阵](eeg_emotion/outputs/figures/trial_level_confusion_matrix.png)


## 6. 讨论

实验结果表明，BD-Conformer 能够比传统手工特征方法更好地利用原始 EEG 的时空信息。window normalization 对跨被试任务较重要，可以缓解不同被试信号尺度不一致的问题。balanced-rank 利用公开类别比例先验，使同一被试内预测类别数与实验设计一致，从而提升预测稳定性。

trial-level 结果需要谨慎解释。训练集 trial 约为 50 秒，可切分为多个 10 秒窗口并进行概率平均；公开测试集每个 trial 只有 10 秒，无法进行相同聚合。因此，84.17% 反映的是训练集 trial-level LOSO 验证能力，而 73.80% 是更保守的窗口级参考。

从规范性角度看，本文没有使用公开测试集标签，也没有根据线上反馈反推标签。公开测试集仅用于最终无标签推理：读取 80 个 trial，输出概率，按每名被试 4/4 先验进行 balanced-rank，生成 Excel submission 文件。所有监督评估指标均来自训练集 LOSO。

## 7. 结论

本文提出了 BD-Conformer + window normalization + balanced-rank 的跨被试 EEG 情绪识别方法。窗口级 full LOSO 达到 73.80%，trial-level 聚合评估达到 84.17%。结果表明，深度时空模型结合公开实验先验能够提升跨被试 EEG 情绪识别性能。考虑到公开测试集每个 trial 仅有 10 秒数据，最终结果解释应同时参考窗口级和 trial-level 两种指标。

## 参考文献

[1] 赛题方. 数据集说明文档f.pdf. 脑机接口赛道赛题四：基于脑电数据的情绪识别算法.

[2] Song, Y., Zheng, Q., Liu, B., & Gao, X. EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 31, 710-719, 2023. DOI: 10.1109/TNSRE.2022.3230250.

[3] Braindecode Documentation. EEGConformer model API. https://braindecode.org/stable/api.html

[4] Patel, P., Raghunandan, R., & Annavarapu, R. N. EEG-based human emotion recognition using entropy as a feature extraction measure. Brain Informatics, 8, 20, 2021.

## 附录：复现实验命令

代码仓库 base URL：<https://github.com/Harry050118/brain_interface.git>

窗口级 balanced-rank full LOSO：

```powershell
D:\Anaconda\envs\eegfm\python.exe eeg_emotion\run_gpu_baselines.py `
  --model bd_conformer `
  --full-loso `
  --epochs 20 `
  --patience 3 `
  --amp `
  --dropout 0.5 `
  --norm-mode window `
  --balanced-rank
```

Trial-level balanced-rank full LOSO：

```powershell
D:\Anaconda\envs\eegfm\python.exe eeg_emotion\run_gpu_baselines.py `
  --model bd_conformer `
  --full-loso `
  --epochs 20 `
  --patience 3 `
  --amp `
  --dropout 0.5 `
  --norm-mode window `
  --trial-balanced-rank
```

生成最终提交文件：

```powershell
D:\Anaconda\envs\eegfm\python.exe eeg_emotion\run_gpu_baselines.py `
  --model bd_conformer `
  --skip-loso `
  --epochs 20 `
  --patience 3 `
  --amp `
  --dropout 0.5 `
  --norm-mode window `
  --trial-balanced-rank `
  --save-submission `
  --output eeg_emotion\outputs\submission_bd_conformer_trial_balanced_rank.xlsx `
  --model-output eeg_emotion\outputs\models\bd_conformer_trial_balanced_rank.pt
```
