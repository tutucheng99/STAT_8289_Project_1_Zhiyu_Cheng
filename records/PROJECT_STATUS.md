# Sepsis Treatment RL Project - Status Report

**Date:** 2024-10-14
**Deadline:** 2024-10-27 (13 days remaining)
**Time Budget:** 2-3 hours/day

---

## Project Goal

Learn optimal and interpretable treatment strategies for septic patients using Reinforcement Learning, comparing offline RL (CQL), online RL (DQN), and baselines (BC, Heuristic).

## Completed Tasks ✓

### Phase 0: Core Infrastructure (Day 1) ✓

1. **Reward Functions Module** (`src/envs/reward_functions.py`)
   - ✓ Simple reward: Terminal ±15 only
   - ✓ Paper reward: SOFA + lactate continuous feedback (Raghu et al. 2017)
   - ✓ Hybrid reward: Scaled intermediate + strong terminal
   - ✓ Function registry and getter

2. **Environment Wrapper** (`src/envs/sepsis_wrapper.py`)
   - ✓ Wraps Gym-Sepsis with custom reward functions
   - ✓ Compatible with d3rlpy and Stable-Baselines3
   - ✓ State tracking for reward computation
   - ✓ Factory function `make_sepsis_env()`

3. **Evaluation Metrics** (`src/evaluation/metrics.py`)
   - ✓ `evaluate_policy()`: Comprehensive policy evaluation
   - ✓ SOFA-stratified metrics (low/medium/high severity)
   - ✓ `print_evaluation_results()`: Pretty printing
   - ✓ `compare_policies()`: Side-by-side comparison

4. **Data Collection**
   - ✓ Offline dataset collected (~20K transitions)
   - Location: `data/offline_dataset.pkl`

---

## Pending Tasks

### Phase 1: Baselines (Days 2-3)

- [ ] Implement random policy evaluation
- [ ] Implement heuristic policy evaluation
- [ ] Create baseline comparison notebook

### Phase 2: Core Algorithms (Days 4-9)

**BC (Behavior Cloning)**
- [ ] Create Colab notebook for BC training
- [ ] Train BC model with simple reward
- [ ] Evaluate on test set

**CQL (Conservative Q-Learning - Offline RL)**
- [ ] Create Colab notebook for CQL training
- [ ] Train CQL model with simple reward
- [ ] Evaluate on test set

**DQN (Deep Q-Network - Online RL)**
- [ ] Create Colab notebook for DQN training
- [ ] Train DQN model with simple reward
- [ ] Evaluate on test set

### Phase 3: Reward Comparison (Day 10)

- [ ] Select best algorithm from Phase 2
- [ ] Train with paper reward function
- [ ] Train with hybrid reward function
- [ ] Compare 3 reward variants

### Phase 4: Analysis & Visualization (Day 11)

- [ ] Create policy heatmap (Figure 1 style)
- [ ] Create mortality vs dosage plot (Figure 2 style)
- [ ] Generate SOFA-stratified analysis
- [ ] Create results comparison table

### Phase 5: Report (Days 12-13)

- [ ] Write paper following JASA template (max 25 pages)
- [ ] Include all figures and tables
- [ ] Finalize code submission

---

## Project Structure

```
project_1/
├── src/
│   ├── envs/
│   │   ├── reward_functions.py       [✓ DONE]
│   │   └── sepsis_wrapper.py         [✓ DONE]
│   ├── evaluation/
│   │   └── metrics.py                [✓ DONE]
│   └── visualization/
│       └── (pending)
├── colab_notebooks/
│   └── (to be created)
├── data/
│   └── offline_dataset.pkl           [✓ DONE]
├── results/
│   ├── models/
│   ├── logs/
│   └── figures/
└── PROJECT_STATUS.md                  [✓ DONE]
```

---

## Experimental Design

### Algorithms to Compare

| Type | Algorithm | Library | Priority |
|------|-----------|---------|----------|
| Baseline | Random | Custom | P0 |
| Baseline | Heuristic | Custom | P0 |
| Baseline | BC | d3rlpy | P0 |
| Offline RL | CQL | d3rlpy | P0 |
| Online RL | DQN | SB3 | P0 |

### Reward Functions

| Name | Description | When to Use |
|------|-------------|-------------|
| Simple | Terminal ±15 only | Phase 2 (all algorithms) |
| Paper | SOFA + lactate feedback | Phase 3 (best algorithm) |
| Hybrid | Scaled intermediate + terminal | Phase 3 (best algorithm) |

### Evaluation Metrics

**Primary:**
- Survival Rate (%)
- Average Return
- SOFA-stratified performance

**Secondary:**
- Episode length
- Policy interpretability (heatmaps)
- Mortality vs dosage deviation

---

## Next Steps (Immediate)

1. **Create Colab Setup Notebook** (1 hour)
   - Environment setup (gym-sepsis, d3rlpy, SB3)
   - Upload data to Colab
   - Test reward functions and wrapper
   - Import existing utility modules

2. **Implement Baseline Evaluations** (1 hour)
   - Random policy
   - Heuristic policy
   - Generate baseline results

3. **Start BC Training** (Tomorrow)
   - Create BC notebook
   - Train on offline data
   - Evaluate and compare to baselines

---

## Key Decisions Made

1. **Tech Stack:** PyTorch ecosystem (d3rlpy + SB3) for better library support
2. **Experiment Scope:** Conservative (6 experiments) to ensure completion
   - 3 algorithms × simple reward
   - 1 best algorithm × 3 rewards
3. **Focus:** Interpretability > Performance > Methodology > Reward Engineering
4. **Compute:** Google Colab with A100 GPU

---

## Risk Mitigation

**Risk 1:** Training takes longer than expected
→ Solution: Have checkpoints, reduce n_episodes if needed

**Risk 2:** Results are not interesting
→ Solution: Focus on interpretability and clinical analysis

**Risk 3:** Time crunch for report writing
→ Solution: Start documenting results as you go

---

## Daily Checklist Template

### Morning (1 hour)
- [ ] Review previous day's results
- [ ] Update PROJECT_STATUS.md
- [ ] Start next experiment

### Evening (1-2 hours)
- [ ] Monitor training progress
- [ ] Document results
- [ ] Prepare next day's tasks

---

## Questions to Address in Report

1. How do offline vs online RL compare in this medical domain?
2. Which reward function leads to better policies?
3. Do learned policies match clinical intuition?
4. How does performance vary by patient severity (SOFA)?
5. Are there safety concerns with the learned policies?

---

## Resources

- **Paper:** Raghu et al. (2017) - Deep RL for Sepsis Treatment
- **Environment:** gym-sepsis (GitHub)
- **d3rlpy Docs:** https://d3rlpy.readthedocs.io/
- **SB3 Docs:** https://stable-baselines3.readthedocs.io/
- **JASA Template:** In `jasa_template/`

---

**Last Updated:** 2024-10-14 (Day 1 - Infrastructure Complete)