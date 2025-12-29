# DRL-for-Vancomycin-PK-PD

This project uses DRL methods to determine the best Vancomycin injection for ICU patients based on MIMIC demo.

## Structure

```text
DRL-for-Vancomycin-PK-PD/
├── data_processing/   # 数据清洗、预处理脚本 (Python)
├── algorithms/        # 核心算法模型、训练脚本 (PyTorch)
├── software/          # 软件交付物 (Web API, Frontend)
│   ├── backend/
│   └── frontend/
├── original_data      # 原始数据
├── .gitignore         # 全局忽略文件
└── README.md          # 整个项目的总导览

# Rawdata 数据说明文档

## 1. 数据集概述

本数据集（`rawdata.xlsx`）为**患者住院期间的时间序列临床原始数据（raw data）**，以 ICU/住院 `stay` 为单位，整合了**实验室检查（lab）**、**生命体征（vitals）**以及**万古霉素（Vancomycin）用药相关信息**。数据主要用于临床状态分析、风险预警建模或药物治疗相关研究。

* **文件格式**：Excel（`.xlsx`）
* **Sheet 数量**：1（Sheet1）
* **样本规模**：13,345 行 × 27 列
* **数据粒度**：事件级（event-level），非固定时间间隔

每一行表示某一患者在相对住院时间轴上的一次事件记录（如一次化验、一次生命体征测量或一次用药相关记录）。

---

## 2. 主键与索引结构

| 字段名          | 含义             | 说明                         |
| ------------ | -------------- | -------------------------- |
| `subject_id` | 患者唯一标识         | 同一患者在不同住院中保持不变             |
| `hadm_id`    | 住院标识           | 一次住院对应一个 `hadm_id`         |
| `stay_id`    | ICU/住院 stay 标识 | 分析与建模的主要单位                 |
| `rel_time`   | 相对时间（小时）       | 相对于 `intime` 的小时数，用于时间序列对齐 |

> **推荐联合主键**：`subject_id + hadm_id + stay_id + rel_time`

---

## 3. 事件类型说明

| 字段名          | 含义     |                                  |
| ------------ | ------ | -------------------------------- |
| `event_type` | 事件类别   | 主要包括：`lab`（实验室检查）、`vitals`（生命体征） |
| `itemid`     | 事件项目编码 | 对应具体化验或体征项目（可能为空）                |

`event_type` 用于区分不同来源的数据，同一时间点可能存在多条不同类型的事件记录。

---

## 4. 万古霉素（Vancomycin）相关变量

以下变量仅在与万古霉素治疗相关的时间段内有值，其余时间可能为空（NaN）：

| 字段名               | 含义       | 单位 / 说明          |
| ----------------- | -------- | ---------------- |
| `totalamount_mg`  | 给药总剂量    | mg               |
| `starttime`       | 给药开始时间   | 绝对时间戳            |
| `vanco_start_rel` | 给药开始相对时间 | 小时（相对 `intime`）  |
| `vanco_end_rel`   | 给药结束相对时间 | 小时               |
| `vanco_level`     | 万古霉素血药浓度 | 通常为 trough level |

---

## 5. 实验室检查指标（Labs）

以下为常见实验室指标字段，通常在 `event_type = 'lab'` 时出现：

| 字段名          | 含义    | 说明      |
| ------------ | ----- | ------- |
| `creatinine` | 肌酐    | 反映肾功能   |
| `wbc`        | 白细胞计数 | 感染/炎症指标 |
| `bun`        | 尿素氮   | 肾功能相关   |
| `charttime`  | 化验时间  | 绝对时间戳   |

---

## 6. 生命体征（Vitals）相关变量

以下变量通常在 `event_type = 'vitals'` 时记录：

| 字段名           | 含义     | 单位    |
| ------------- | ------ | ----- |
| `vitaltime`   | 体征测量时间 | 绝对时间戳 |
| `heart_rate`  | 心率     | 次 / 分 |
| `sbp`         | 收缩压    | mmHg  |
| `temperature` | 体温     | ℃     |

---

## 7. 生命体征预警标签（Warnings）

为便于下游风险建模，数据中已包含基于阈值规则生成的二值预警变量：

| 字段名         | 含义     | 取值说明          |
| ----------- | ------ | ------------- |
| `hr_warn`   | 心率异常预警 | 1 = 异常，0 = 正常 |
| `sbp_warn`  | 血压异常预警 | 1 = 异常，0 = 正常 |
| `temp_warn` | 体温异常预警 | 1 = 异常，0 = 正常 |

> 具体阈值规则需结合实验或论文方法部分进一步说明。

---

## 8. 患者基本信息

| 字段名             | 含义 | 说明     |
| --------------- | -- | ------ |
| `gender`        | 性别 | M / F  |
| `anchor_age`    | 年龄 | 脱敏后的年龄 |
| `patientweight` | 体重 | kg     |

---

## 9. 住院与 ICU 时间信息

| 字段名       | 含义            | 说明     |
| --------- | ------------- | ------ |
| `intime`  | 入院 / 入 ICU 时间 | 时间序列起点 |
| `outtime` | 出院 / 出 ICU 时间 | 可能为空   |

所有相对时间变量（`rel_time`, `vanco_start_rel`, `vanco_end_rel`）均以 `intime` 为参考基准计算。

---

## 10. 缺失值说明

* 不同事件类型只填充与自身相关的字段，其余字段为空（NaN）
* 用药与化验数据在时间轴上高度不规则，属于典型 **稀疏时间序列数据**
* 使用前建议按 `stay_id` 对数据进行重采样、对齐或插值处理（视建模方法而定）

---

## 11. 使用建议

* 建模前建议先按 `stay_id` 分组
* 使用 `rel_time` 作为统一时间轴
* 根据 `event_type` 拆分或融合 lab / vitals / drug 信息
* 若用于论文中的 **Raw Data 描述**，可重点强调：

  * 原始性（未经聚合）
  * 时间不规则性
  * 多模态临床信息融合

---

**文档类型**：Raw Data Description / Data Dictionary
