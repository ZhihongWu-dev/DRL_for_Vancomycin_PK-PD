# READY 数据集说明文档（中文版）

## 一、数据集概述

**READY** 数据集用于序列决策与强化学习任务，主要面向临床用药管理与患者状态建模场景。数据以 **4 小时（4-hour）** 为时间步，对单个进入 ICU病例（stay）进行时间序列建模，适用于策略学习、奖励建模以及离线强化学习（Offline Reinforcement Learning）。

---

## 二、数据结构

- **行（Rows）**：单个病人在某一 4 小时时间步的观测记录  
- **列（Columns）**：病人状态、给药行为、实验室指标及奖励信号  
- **主键（Primary Keys）**：`stay_id` + `step_4hr`

---

## 三、字段说明

### 1. 基本标识

- **stay_id**：病人一次进入 ICU ICU记录的唯一标识符，用于区分不同 episode。
- **step_4hr**：以 4 小时为单位的时间步编号（从 1 开始）。

---

### 2. 行为变量（Action）

- **totalamount_mg**：当前时间步内的给药总剂量（单位：mg），通常作为强化学习中的动作（action）。

---

### 3. 药物与实验室指标

- **vanco_level (ug/mL)**：药物血药浓度（μg/mL）。
- **creatinine (mg/dL)**：肌酐水平，用于反映肾功能状态。
- **wbc (K/uL)**：白细胞计数，反映感染或炎症情况。
- **bun (mg/dL)**：血尿素氮水平。

---

### 4. 生命体征

- **temperature**：体温（摄氏度）。
- **sbp**：收缩压（mmHg）。
- **heart_rate**：心率（次/分钟）。

---

### 5. 缺失值标记

以下字段为二值变量，`1` 表示该时间步该指标缺失，`0` 表示观测到真实值：

- **creatinine(mg/dL)_is_missing**
- **wbc(K/uL)_is_missing**
- **bun(mg/dL)_is_missing**

---

### 6. 差分特征

- **cre_diff**：当前时间步与上一时间步肌酐值的差值。
- **wbc_diff**：当前时间步与上一时间步白细胞计数的差值。

---

### 7. 风险与奖励信号

- **warning**：风险或异常状态标记（0/1），可用于安全约束或终止条件。
- **step_reward**：当前时间步的即时奖励信号，用于强化学习训练。

---

## 四、典型用途

- 离线强化学习（Offline RL）
- 给药策略优化
- 临床决策支持系统
- 时间序列建模与患者状态预测

---

## 五、建模建议

- 使用 `stay_id` 构建 episode / trajectory
- 将 `totalamount_mg` 视为 action
- 其余生理与实验室指标作为 state
- 使用 `step_reward` 作为 reward
- 可结合 `warning` 构建安全强化学习（Safe RL）

---

## 六、备注

- 数据已按时间顺序排序
- 差分特征在首个时间步可能为 NaN
- 可根据研究需要进行标准化或离散化处理


---

# READY Dataset Documentation (English Version)

## 1. Overview

The **READY** dataset is designed for sequential decision-making and reinforcement learning tasks, particularly in clinical medication management and patient state modeling. Data are organized in **4-hour time steps**, enabling trajectory-based modeling for eICU stay The dataset can be directly used for policy learning, reward modeling, and offline reinforcement learning.

---

## 2. Data Structure

- **Rows**: One patient observation at a specific 4-hour time step  
- **Columns**: Patient states, medication actions, laboratory measurements, and reward signals  
- **Primary Keys**: `stay_id` + `step_4hr`

---

## 3. Column Descriptions

### 3.1 Identifiers

- **stay_id**: Unique identifier for a single IICU stay to define an episode.
- **step_4hr**: Time step index in 4-hour intervals (starting from 1).

---

### 3.2 Action Variable

- **totalamount_mg**: Total administered drug dose (mg) at the current time step, typically treated as the action in reinforcement learning.

---

### 3.3 Drug and Laboratory Measurements

- **vanco_level (ug/mL)**: Serum drug concentration (μg/mL).
- **creatinine (mg/dL)**: Creatinine level, reflecting renal function.
- **wbc (K/uL)**: White blood cell count, indicating infection or inflammation.
- **bun (mg/dL)**: Blood urea nitrogen level.

---

### 3.4 Vital Signs

- **temperature**: Body temperature (°C).
- **sbp**: Systolic blood pressure (mmHg).
- **heart_rate**: Heart rate (beats per minute).

---

### 3.5 Missing Value Indicators

The following binary variables indicate whether a measurement is missing (`1`) or observed (`0`) at the given time step:

- **creatinine(mg/dL)_is_missing**
- **wbc(K/uL)_is_missing**
- **bun(mg/dL)_is_missing**

---

### 3.6 Delta / Trend Features

- **cre_diff**: Difference in creatinine level compared to the previous time step.
- **wbc_diff**: Difference in white blood cell count compared to the previous time step.

---

### 3.7 Risk and Reward Signals

- **warning**: Risk or abnormal state indicator (0/1), which can be used for safety constraints or termination conditions.
- **step_reward**: Immediate reward at the current time step, used for reinforcement learning training.

---

## 4. Typical Use Cases

- Offline reinforcement learning
- Medication dosing policy optimization
- Clinical decision support systems
- Time-series modeling and patient state prediction

---

## 5. Modeling Notes

- Use `stay_id` to construct episodes or trajectories
- Treat `totalamount_mg` as the action variable
- Use physiological and laboratory variables as state features
- Use `step_reward` as the reward signal
- `warning` can be incorporated for Safe Reinforcement Learning

---

## 6. Notes

- Data are temporally ordered
- Delta features may be NaN at the first time step
- Normalization or discretization can be applied as required

---

**End of READY Dataset Documentation**

