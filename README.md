# DRL-for-Vancomycin-PK-PD

This project uses DRL methods to determine the best Vancomycin injection for ICU patients based on MIMIC demo.

## Structure

```text
DRL-for-Vancomycin-PK-PD/
├── data_processing/   # 数据清洗、预处理脚本 (Python, SQL)
├── algorithms/        # 核心算法模型、训练脚本 (PyTorch)
├── software/          # 软件交付物 (Web API, Frontend)
│   ├── backend/
│   └── frontend/
├── docs/              # 跨模块的文档说明
├── .gitignore         # 全局忽略文件
└── README.md          # 整个项目的总导览
