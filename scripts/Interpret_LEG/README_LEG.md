# LEG Analysis Scripts

Two scripts for Linearly Estimated Gradient (LEG) analysis of RL policies in sepsis treatment.

## 📁 Scripts

### 1. `leg_analysis_offline.py` - For Offline RL (d3rlpy)
Analyzes models trained with d3rlpy (BC, CQL, offline DQN).

### 2. `leg_analysis_online.py` - For Online RL (Stable-Baselines3)
Analyzes models trained with SB3 (DQN, A2C, PPO).

---

## 🚀 Usage

### Online RL (Your DQN Model)

```bash
# Basic usage (default: 10 states)
python scripts/leg_analysis_online.py

# Custom number of states
python scripts/leg_analysis_online.py --n_states 20

# Full customization
python scripts/leg_analysis_online.py \
    --model results/models/dqn_simple_reward.zip \
    --n_states 20 \
    --n_samples 1000 \
    --output_dir results/figures/leg_online
```

### Offline RL (BC/CQL Models)

```bash
# Analyze BC model
python scripts/leg_analysis_offline.py \
    --model results/models/bc_simple_reward.d3 \
    --n_states 10

# Analyze CQL model
python scripts/leg_analysis_offline.py \
    --model results/models/cql_simple_reward.d3 \
    --n_states 10
```

---

## 📊 Output

Each script generates 2 types of visualizations per state:

### 1. Detailed Analysis (`analysis_state_N.png`)
- **Top Features Bar Chart**: LEG saliency scores for selected action
- **Feature Values**: Current state values for top features
- **Q-Values**: Q-values across all 24 actions

### 2. Saliency Heatmap (`saliency_state_N.png`)
- **Rows**: Top-15 most important features
- **Columns**: Representative actions (0, 6, 12, 18, 23)
- **Colors**: Red/Blue indicate positive/negative gradient
- **Numbers**: Exact saliency scores

**Output Location:**
- Online: `results/figures/leg_online/dqn_simple_reward/`
- Offline: `results/figures/leg_offline/<model_name>/`

---

## 🎯 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | (varies) | Path to trained model (.zip for SB3, .d3 for d3rlpy) |
| `--n_states` | 10 | Number of states to analyze |
| `--n_samples` | 1000 | Perturbation samples for gradient estimation |
| `--output_dir` | `results/figures/leg_*` | Output directory for figures |

---

## 🔬 What is LEG?

**Linearly Estimated Gradient (LEG)** estimates how much each feature influences Q-values:

### Formula
```
γ̂(π, s₀, F) = Σ⁻¹ (1/n) Σᵢ(ŷᵢZᵢ)
```

Where:
- `Zᵢ`: Perturbation vector (random noise added to features)
- `ŷᵢ`: Change in Q-value due to perturbation
- `Σ`: Covariance matrix of perturbations
- `γ̂`: Estimated gradient (saliency score)

### Interpretation
- **Positive score**: Increasing feature → higher Q-value
- **Negative score**: Increasing feature → lower Q-value
- **Large magnitude**: Feature strongly influences decision

---

## 💡 Example Results

### State 1: No Treatment (Action 0)
```
Selected Action: 0 (IV=0, VP=0)
Top 5 Features:
  1. INR: score=-0.600 (high INR → don't select Action 0)
  2. RespRate: score=0.459 (high RespRate → select Action 0)
  3. LACTATE: score=0.417 (high Lactate → select Action 0)
```

**Interpretation**: DQN chooses no treatment when INR is low but RespRate/Lactate are elevated.

### State 2: Aggressive Treatment (Action 16)
```
Selected Action: 16 (IV=3, VP=1)
Top 5 Features:
  1. PLATELET: score=0.029
  2. BILIRUBIN: score=-0.027
```

**Interpretation**: Smaller saliency scores → decision less dependent on single features, more holistic.

---

## 📈 Performance Tips

### Fast Testing (3-5 minutes)
```bash
python scripts/leg_analysis_online.py --n_states 3 --n_samples 100
```

### Production Quality (30-60 minutes)
```bash
python scripts/leg_analysis_online.py --n_states 20 --n_samples 1000
```

### Memory Considerations
- Each state: ~100 MB memory (1000 samples × 46 features × 24 actions)
- Reduce `--n_samples` if running out of memory

---

## 🆚 Comparison: Offline vs Online

| Aspect | Offline Script | Online Script |
|--------|----------------|---------------|
| **Library** | d3rlpy | Stable-Baselines3 |
| **Models** | BC, CQL, offline DQN | DQN, A2C, PPO |
| **File Format** | `.d3` | `.zip` |
| **Q-values** | `model.predict_value()` | `model.q_net()` |
| **Use Case** | Your BC/CQL models | **Your trained DQN (96.5%)** |

---

## 🐛 Troubleshooting

### Issue: "Model not found"
```bash
# Check model exists
ls results/models/

# Use absolute path
python scripts/leg_analysis_online.py --model "C:/full/path/to/model.zip"
```

### Issue: Out of memory
```bash
# Reduce samples
python scripts/leg_analysis_online.py --n_samples 500

# Analyze fewer states
python scripts/leg_analysis_online.py --n_states 5
```

### Issue: Slow execution
- LEG analysis is computationally intensive
- 1000 samples × 10 states = ~10,000 forward passes through Q-network
- Expected time: 3-5 minutes per state

---

## 📚 References

- **LEG Paper**: Greydanus et al. (2018) "Visualizing and Understanding Atari Agents"
- **Your DQN Performance**: 96.5% survival, 93.5% on high-SOFA patients
- **Use Case**: Understand why DQN outperforms BC/CQL on critical patients

---

## ✅ Next Steps

1. **Run full analysis**:
   ```bash
   python scripts/leg_analysis_online.py --n_states 20
   ```

2. **Compare with BC/CQL**:
   ```bash
   python scripts/leg_analysis_offline.py --model results/models/bc_simple_reward.d3
   python scripts/leg_analysis_offline.py --model results/models/cql_simple_reward.d3
   ```

3. **Analyze findings**:
   - Which features does DQN rely on most?
   - How does it differ from BC/CQL?
   - Why does it excel on high-SOFA patients?

4. **Add to paper**:
   - Include saliency heatmaps in Results section
   - Discuss feature importance in Discussion
   - Explain DQN's superior performance on critical patients
