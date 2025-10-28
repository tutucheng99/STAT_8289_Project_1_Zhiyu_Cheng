# 🏠 本地运行指南 - Python 脚本版

**推荐方案：本地 CPU 运行**

根据 `train_offline_policy.ipynb` 的测试，CPU 性能完全够用！

---

## ⚡ 快速开始

### 1. 创建虚拟环境

```bash
# 进入项目目录
cd "C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1"

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 或者激活（Mac/Linux）
source .venv/bin/activate
```

### 2. 安装依赖

```bash
# 升级 pip
python -m pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt

# 安装 gym-sepsis（本地安装）
pip install -e gym-sepsis
```

### 3. 验证安装

```bash
python scripts/00_test_installation.py
```

### 4. 运行实验

```bash
# 运行所有实验（自动化）
python run_experiments.py --all

# 或者单独运行各个阶段
python run_experiments.py --baseline    # Step 1: 基线评估
python run_experiments.py --train-rl    # Step 2: 训练 RL 算法
python run_experiments.py --reward-comp # Step 3: 奖励函数对比
python run_experiments.py --visualize   # Step 4: 生成图表
python run_experiments.py --analyze     # Step 5: 最终分析
```

---

## 📁 项目结构

```
project_1/
├── .venv/                      # 虚拟环境（自动生成）
├── gym-sepsis/                 # Sepsis 环境
├── src/                        # 源代码
│   ├── envs/                   # 环境包装器
│   ├── evaluation/             # 评估模块
│   └── visualization/          # 可视化模块
├── scripts/                    # 实验脚本（新建）
│   ├── 00_test_installation.py
│   ├── 01_baseline_evaluation.py
│   ├── 02_train_bc.py
│   ├── 03_train_cql.py
│   ├── 04_train_dqn.py
│   ├── 05_reward_comparison.py
│   ├── 06_visualization.py
│   └── 07_final_analysis.py
├── data/                       # 数据
│   └── offline_dataset.pkl
├── results/                    # 实验结果（自动生成）
│   ├── models/                 # 训练的模型
│   ├── figures/                # 生成的图表
│   └── logs/                   # 训练日志
├── run_experiments.py          # 主运行脚本
├── requirements.txt            # Python 依赖
└── LOCAL_SETUP.md             # 本文档
```

---

## 🎯 运行时间估计（CPU）

根据 `train_offline_policy.ipynb` 的实测数据：

| 实验 | 预计时间 | 说明 |
|------|---------|------|
| Baseline | 10-15分钟 | 评估随机和启发式策略 |
| BC Training | 30分钟 | 行为克隆 |
| CQL Training | 1-2小时 | 保守Q学习 |
| DQN Training | 1-2小时 | 深度Q网络 |
| Reward Comparison | 2-3小时 | 测试3个奖励函数 |
| Visualization | 5-10分钟 | 生成图表 |
| Analysis | 5分钟 | 最终分析 |
| **总计** | **6-9小时** | CPU完全够用！ |

---

## 🔧 高级用法

### 只运行特定算法

```bash
# 只训练 BC
python scripts/02_train_bc.py

# 只训练 CQL
python scripts/03_train_cql.py --epochs 100 --batch-size 256

# 只训练 DQN
python scripts/04_train_dqn.py --steps 200000
```

### 自定义参数

```python
# 在脚本中修改参数
python scripts/02_train_bc.py \
    --epochs 200 \
    --batch-size 512 \
    --learning-rate 0.0001 \
    --reward-fn simple
```

### 恢复中断的训练

```bash
# 从检查点恢复
python scripts/03_train_cql.py --resume results/models/cql_checkpoint.pt
```

---

## 📊 查看结果

训练完成后，结果保存在 `results/` 目录：

```
results/
├── baseline_results.pkl         # 基线结果
├── bc_results.pkl              # BC 结果
├── cql_results.pkl             # CQL 结果
├── dqn_results.pkl             # DQN 结果
├── figures/                    # 图表
│   ├── baseline_comparison.png
│   ├── algorithm_comparison.png
│   ├── reward_comparison.png
│   └── policy_heatmaps.png
├── models/                     # 训练的模型
│   ├── bc_simple_reward.pt
│   ├── cql_simple_reward.pt
│   └── dqn_simple_reward.zip
└── logs/                       # 训练日志
    └── training.log
```

---

## 🐛 常见问题

### Q1: TensorFlow 版本冲突
**A:** 本地环境使用最新版本即可，不需要降级：
```bash
pip install tensorflow  # 自动选择最新兼容版本
```

### Q2: gym-sepsis 导入错误
**A:** 确保已经安装：
```bash
pip install -e gym-sepsis
```

### Q3: CUDA 不可用
**A:** 没问题！CPU 完全够用，不需要 GPU：
```python
# 脚本会自动检测并使用 CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Q4: 内存不足
**A:** 减小 batch size：
```bash
python scripts/03_train_cql.py --batch-size 128
```

---

## 💡 优化建议

### 1. 并行运行（如果有多核CPU）

```bash
# 在不同终端同时运行
# Terminal 1
python scripts/02_train_bc.py

# Terminal 2
python scripts/03_train_cql.py

# Terminal 3
python scripts/04_train_dqn.py
```

### 2. 后台运行

```bash
# Windows (PowerShell)
Start-Job -ScriptBlock { python run_experiments.py --all }

# Mac/Linux
nohup python run_experiments.py --all > output.log 2>&1 &
```

### 3. 监控进度

```bash
# 实时查看日志
tail -f results/logs/training.log

# 或者在 Python 中使用 tqdm 进度条
```

---

## 🎓 开发模式

如果要修改代码并测试：

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 启动 Jupyter Lab（可选）
jupyter lab

# 或者直接用 VS Code / PyCharm 打开项目
```

---

## 📝 环境变量（可选）

创建 `.env` 文件配置路径：

```bash
# .env
PROJECT_ROOT=C:\Users\tutu9\OneDrive\桌面\EVERYTHING\learning\GWU\Fall 2025\Stats 8289\project_1
RESULTS_DIR=results
DATA_DIR=data
N_JOBS=4  # 并行任务数
```

---

## ✅ 检查清单

在运行实验前，确保：

- [ ] 虚拟环境已创建并激活
- [ ] 所有依赖已安装（`pip install -r requirements.txt`）
- [ ] gym-sepsis 已安装（`pip install -e gym-sepsis`）
- [ ] 数据文件存在（`data/offline_dataset.pkl`）
- [ ] 有足够的磁盘空间（~2GB for models and results）

---

## 🚀 开始实验

```bash
# 一键运行所有实验
python run_experiments.py --all

# 大约 6-9 小时后，你会得到：
# - 所有模型训练完成
# - 图表自动生成
# - 结果汇总报告
# - 准备好写论文的数据！
```

---

**Good Luck! 🎉**

本地运行比 Colab 更简单、更稳定、更快速！
