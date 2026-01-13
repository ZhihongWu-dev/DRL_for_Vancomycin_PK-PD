# IQL (Implicit Q-Learning) — Vancomycin给药优化

完整的IQL离线强化学习实现，用于万古霉素(Vancomycin)个性化给药优化。

## 🎯 项目成果

### ✅ 核心实现
- **数据处理** (`dataset.py`): 2102个有效临床转移
- **神经网络** (`models.py`): Q/V/Policy网络
- **训练系统** (`train_iql.py`): 完整管道+检查点+TensorBoard
- **离线评估** (`evaluate_iql.py`): 价值函数和策略评估
- **可视化分析** (`analysis.ipynb`): 交互式策略分析

### 📊 最优结果 (exp_conservative)

| 指标 | 初始 → 最终 | 改进 |
|------|-----------|------|
| Q损失 | 44.3 → 1.51 | ↓ -96.6% |
| V损失 | 12.6 → 0.16 | ↓ -98.7% |
| 贪心策略Q值 | - | -91.52 |

**最佳检查点**: `runs/exp_conservative/ckpt_step3000.pt` (学习率0.00003, gamma=0.90)

## 🚀 快速开始

### 训练
```bash
python -m algorithms.iql.train_iql --config configs/iql_conservative.yaml
```

### 评估
```bash
python -m algorithms.iql.evaluate_iql \
  --checkpoint algorithms/iql/runs/exp_conservative/ckpt_step3000.pt \
  --config configs/iql_conservative.yaml \
  --output eval_results.json
```

### 可视化
打开 `analysis.ipynb`:
- Q/V值分布与统计
- 不同状态的推荐剂量曲线
- 临床特征敏感性分析
- 策略与行为对比

### TensorBoard
```bash
tensorboard --logdir algorithms/iql/runs --port 6006
```

## 📁 文件说明

- Run tests for the IQL package:
  - `python -m algorithms.iql.run_tests`  (pytest will run tests under `algorithms/iql/tests`)

- Start TensorBoard for IQL runs:
  - `python -m algorithms.iql.run_tensorboard --logdir algorithms/iql/runs --port 6006`

Notes:
- Test discovery is configured in `pytest.ini` to only run tests under `algorithms/iql/tests`.
- Training outputs (checkpoints, tensorboard logs) default to `algorithms/iql/runs/` and are ignored by `.gitignore`.
