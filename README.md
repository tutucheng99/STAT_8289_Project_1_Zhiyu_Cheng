# Deep Reinforcement Learning for Sepsis Treatment

**Course:** STAT 8289 - Reinforcement Learning
**Due Date:** October 27, 2024
**Team:** Individual Project

---

## Project Overview

This project applies reinforcement learning to learn optimal treatment strategies for septic patients in the ICU. We compare:
- **Offline RL:** CQL (Conservative Q-Learning)
- **Online RL:** DQN (Deep Q-Network)
- **Baselines:** Random, Heuristic, Behavior Cloning

### Key Features
- 3 reward function variants (Simple, Paper, Hybrid)
- SOFA-stratified performance analysis
- Clinical interpretability focus
- Comprehensive evaluation metrics

---

## Quick Start

### For Google Colab (Recommended)

1. **Read the Guide:**
   ```
   Open: COLAB_GUIDE.md
   ```

2. **Upload to Google Drive:**
   - Upload entire `project_1/` folder
   - Or create zip: `sepsis_rl_project.zip`

3. **Run Notebooks:**
   ```
   Start with: colab_notebooks/01_Setup_and_Baseline_Evaluation.ipynb
   ```

### For Local Development

1. **Setup Environment:**
   ```bash
   conda create -n sepsis_rl python=3.10
   conda activate sepsis_rl
   pip install -r requirements.txt  # (to be created)
   ```

2. **Test Modules:**
   ```bash
   python src/envs/reward_functions.py
   python src/evaluation/metrics.py
   ```

---

## Project Structure

```
project_1/
├── README.md                    # This file
├── PROJECT_STATUS.md            # Detailed progress tracker
├── COLAB_GUIDE.md              # Google Colab workflow guide
│
├── prompts/                     # Project requirements
│   ├── Deep Reinforcement Learning for Sepsis Treatment.pdf
│   └── STAT 8289 RL project.pdf
│
├── gym-sepsis/                  # Sepsis simulation environment
│   └── gym_sepsis/
│       └── envs/
│           └── sepsis_env.py
│
├── src/                         # Source code [✓ COMPLETED]
│   ├── envs/
│   │   ├── reward_functions.py      # 3 reward variants
│   │   └── sepsis_wrapper.py        # Environment wrapper
│   ├── evaluation/
│   │   └── metrics.py               # Evaluation tools
│   ├── data/
│   │   └── collect_data.py          # Data collection script
│   └── visualization/
│       └── (to be added)
│
├── data/                        # Datasets
│   └── offline_dataset.pkl          # ~20K transitions
│
├── colab_notebooks/             # Jupyter notebooks for Colab
│   ├── 01_Setup_and_Baseline_Evaluation.ipynb  [✓ DONE]
│   ├── 02_Train_BC.ipynb                        [TODO]
│   ├── 03_Train_CQL.ipynb                       [TODO]
│   ├── 04_Train_DQN.ipynb                       [TODO]
│   └── ...
│
├── results/                     # Experiment results
│   ├── models/                  # Trained models
│   ├── logs/                    # Training logs
│   └── figures/                 # Plots and visualizations
│
└── jasa_template/               # Paper template
```

---

## Implementation Progress

### ✅ Completed (Day 1)

**Core Modules (src/):**
- [x] Reward functions module (3 variants)
- [x] Environment wrapper (reward switching)
- [x] Evaluation metrics (SOFA-stratified)
- [x] Random policy
- [x] Heuristic policy

**Infrastructure:**
- [x] Data collection (~20K transitions)
- [x] Project documentation
- [x] Colab notebook #1 (Setup & Baselines)

**Total:** 6/20 tasks completed (30%)

### 🔄 In Progress

None currently - ready to start training!

### 📋 Next Steps (Days 2-3)

**Priority 1:**
- [ ] Create BC training notebook
- [ ] Train BC model
- [ ] Evaluate BC performance

**Priority 2:**
- [ ] Create CQL training notebook
- [ ] Train CQL model
- [ ] Compare to BC

---

## Algorithms

### 1. Behavior Cloning (BC)
- **Type:** Supervised learning baseline
- **Library:** d3rlpy
- **Data:** Offline dataset
- **Expected Time:** ~30 minutes
- **Expected Survival:** 80-85%

### 2. Conservative Q-Learning (CQL)
- **Type:** Offline RL
- **Library:** d3rlpy
- **Data:** Offline dataset
- **Expected Time:** 1-2 hours
- **Expected Survival:** 85-90%

### 3. Deep Q-Network (DQN)
- **Type:** Online RL
- **Library:** Stable-Baselines3
- **Data:** Environment interaction
- **Expected Time:** 1-2 hours
- **Expected Survival:** 85-90%

---

## Reward Functions

### Simple (Default for Phase 1)
```python
# Intermediate: 0
# Terminal: ±15
```

### Paper (Raghu et al. 2017)
```python
# Intermediate: SOFA + lactate feedback
r = C0·1(SOFA unchanged) + C1·ΔS OFA + C2·tanh(Δlactate)
# Terminal: ±15
```

### Hybrid
```python
# Intermediate: 0.1 × paper_reward
# Terminal: ±15
```

---

## Evaluation Metrics

### Primary
- Survival Rate (%)
- Average Return
- Episode Length

### Secondary
- **SOFA-Stratified:**
  - Low SOFA (<5)
  - Medium SOFA (5-15)
  - High SOFA (>15)

### Clinical Interpretability
- Policy heatmaps (action distribution)
- Mortality vs dosage deviation
- Treatment pattern analysis

---

## Timeline

| Days | Phase | Tasks | Status |
|------|-------|-------|--------|
| 1 | Infrastructure | Core modules + Baselines | ✅ Done |
| 2-3 | BC Training | Notebook + training + eval | 📋 Next |
| 4-5 | CQL Training | Notebook + training + eval | ⏳ Pending |
| 6-7 | DQN Training | Notebook + training + eval | ⏳ Pending |
| 8-9 | Phase 2 | Compare algorithms | ⏳ Pending |
| 10 | Reward Study | Test 3 reward variants | ⏳ Pending |
| 11 | Visualization | Create all figures | ⏳ Pending |
| 12-13 | Report | Write paper (JASA format) | ⏳ Pending |

**Current Status:** Day 1 Complete, On Schedule

---

## Key Files Reference

### Configuration
- Feature indices: `src/envs/reward_functions.py:19-20`
- Action space: gym-sepsis uses 24 actions (0-23)
- State space: 46-dimensional continuous

### Dataset
- Location: `data/offline_dataset.pkl`
- Size: ~20,000 transitions
- Episodes: ~2,100
- Collected with: Heuristic policy

### Models (to be trained)
- BC: `results/models/bc_simple_reward.pt`
- CQL: `results/models/cql_simple_reward.pt`
- DQN: `results/models/dqn_simple_reward.pt`

---

## Dependencies

### Core
- Python 3.10
- NumPy, Pandas, Matplotlib, Seaborn

### RL Libraries
- gym==0.21.0 (for gym-sepsis compatibility)
- d3rlpy (offline RL)
- stable-baselines3 (online RL)

### Deep Learning
- PyTorch (for d3rlpy, SB3)
- TensorFlow==2.12.0 (for gym-sepsis)

### Compute
- GPU: NVIDIA A100 (Google Colab)
- Memory: ~40GB VRAM recommended
- Storage: ~5GB for models and results

---

## References

1. **Original Paper:**
   - Raghu et al. (2017). "Deep Reinforcement Learning for Sepsis Treatment"
   - NIPS 2017

2. **Environment:**
   - gym-sepsis: https://github.com/...

3. **Libraries:**
   - d3rlpy: https://d3rlpy.readthedocs.io/
   - Stable-Baselines3: https://stable-baselines3.readthedocs.io/

4. **Dataset:**
   - MIMIC-III: https://mimic.mit.edu/

---

## Citation

```bibtex
@article{raghu2017deep,
  title={Deep reinforcement learning for sepsis treatment},
  author={Raghu, Aniruddh and Komorowski, Matthieu and Ahmed, Imran and Celi, Leo and Szolovits, Peter and Ghassemi, Marzyeh},
  journal={arXiv preprint arXiv:1711.09602},
  year={2017}
}
```

---

## Contact

**Student:** [Your Name]
**Course:** STAT 8289 - Reinforcement Learning
**Institution:** George Washington University
**Semester:** Fall 2024

---

## License

Academic use only. Based on gym-sepsis and MIMIC-III data usage agreements.

---

**Last Updated:** 2024-10-14 (Day 1)
**Next Milestone:** BC Training (Day 2)
**Progress:** 30% Complete (6/20 tasks)
