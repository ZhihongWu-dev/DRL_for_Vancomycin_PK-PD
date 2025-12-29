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
├── intermediate_data  # 中间数据
├── .gitignore         # 全局忽略文件
└── README.md          # 整个项目的总导览

