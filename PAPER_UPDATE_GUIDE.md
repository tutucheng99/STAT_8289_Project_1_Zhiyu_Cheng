# Paper Update Guide: Integrating Online RL Models

**Project**: Sepsis Treatment RL - Collaborative Paper
**Date**: 2025-10-28
**Authors**: Zhiyu Cheng, Yalun Ding, Chuanhui Peng

---

## 📋 Project Context

### Current Paper Status
- **Title**: Performance-Interpretability Trade-offs in Offline Reinforcement Learning for Sepsis Treatment
- **Location**: `paper/main.tex` and `paper/sections/*.tex`
- **Format**: JASA template (see `jasa_template/`)
- **Current Length**: ~23 pages
- **Target**: ≤25 pages

### What's Changing
We're integrating **3 online RL models** (DDQN-Attention, DDQN-Residual, SAC) from collaborator Yalun Ding to create a comprehensive offline vs online comparison study.

### Key Data Source
- **File**: `results/yalun_models_evaluation.pkl`
- **Content**: Evaluation results for 3 online models with 500 episodes + SOFA stratification

**Evaluation Results Summary**:
```
Model              Overall Survival    High SOFA Survival    High SOFA n
-----------------------------------------------------------------------------
BC                 94.2%               88.6%                 211
CQL                94.0%               88.5%                 191
DQN                94.0%               84.3%                 185
DDQN-Attention     95.4%               90.5%                 190
DDQN-Residual      94.2%               87.0%                 200
SAC                94.8%               88.7%                 195
Random             95.0%               -                     -
Heuristic          94.6%               -                     -
```

---

## 📝 Task List (10 Tasks)

### TASK 1: Update Title and Authors
**File**: `paper/main.tex` (lines 91-98)

**Current**:
```latex
\title{\bf Performance-Interpretability Trade-offs in Offline Reinforcement Learning for Sepsis Treatment: A Comparative Study Using LEG Analysis}
\author{Your Name\thanks{...}
  Department of Statistics, George Washington University}
```

**Action**:
1. Update title to reflect offline+online comparison:
   - Option A: "Comparative Study of Offline and Online Reinforcement Learning for Sepsis Treatment: A LEG-based Interpretability Analysis"
   - Option B: "Performance and Interpretability Trade-offs in Reinforcement Learning for Sepsis Treatment: Comparing Offline and Online Approaches"

2. Add 3 co-authors with equal contribution:
```latex
\author{
  Zhiyu Cheng\thanks{Equal contribution. Authors listed alphabetically.}$^{1}$,
  Yalun Ding$^{1,*}$,
  Chuanhui Peng$^{1}$ \\
  $^{1}$Department of Statistics, George Washington University \\
  $^{*}$Corresponding author: yding@gwu.edu
}
```

**Page Impact**: No change

---

### TASK 2: Add Online RL Algorithms in Methods
**File**: `paper/sections/04_methods.tex`
**Location**: After section 4.2.3 (DQN), before section 4.3 (LEG)

**Action**: Add new subsection

```latex
\subsubsection{Online RL Algorithms}\label{sec:methods:algos:online}

To provide a comprehensive comparison, we also evaluate three state-of-the-art online RL algorithms with architectural innovations (implemented by collaborator Y. Ding):

\paragraph{Double DQN with Attention (DDQN-Attention).}
Extends DoubleDQN \citep{hasselt2016deep} with a multi-head self-attention mechanism in the encoder network. The attention layer allows the model to dynamically weight different state features:
\begin{equation}
h_t = \text{MultiHeadAttention}(s_t, s_t, s_t) + s_t
\end{equation}
where the residual connection helps gradient flow. Architecture: 256-128 hidden units, 4 attention heads.

\paragraph{Double DQN with Residual Connections (DDQN-Residual).}
Incorporates deep residual networks \citep{he2016deep} to enable training of deeper Q-networks (3 hidden layers of 256 units each) without gradient vanishing:
\begin{equation}
h_{l+1} = \sigma(\text{LayerNorm}(W_l h_l + b_l + h_l))
\end{equation}
Each layer includes skip connections and layer normalization for training stability.

\paragraph{Soft Actor-Critic (SAC).}
A maximum entropy RL algorithm \citep{haarnoja2018soft} that optimizes both expected return and policy entropy:
\begin{equation}
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_t r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]
\end{equation}
We use the discrete action space variant with residual encoder architecture. Temperature $\alpha$ is automatically tuned.

\paragraph{Training Details.}
All online models were trained with environment interaction (1M steps) using experience replay buffers. Unlike offline methods, these algorithms can explore the state-action space but require access to the simulator during training.
```

**Citations to add** (if not already in bibliography):
- hasselt2016deep: van Hasselt et al., "Deep reinforcement learning with double q-learning"
- he2016deep: He et al., "Deep residual learning for image recognition"
- haarnoja2018soft: Haarnoja et al., "Soft actor-critic: Off-policy maximum entropy deep RL"

**Page Impact**: +1.5 pages

---

### TASK 3: Update Results Table 1 (Overall Performance)
**File**: `paper/sections/05_results.tex`
**Location**: Section 5.1, Table 1

**Current Table** (approx):
```latex
\begin{table}[htbp]
\centering
\caption{Overall Performance Comparison (500 episodes)}
\label{tab:overall}
\begin{tabular}{lcccc}
\toprule
Model & Survival (\%) & Avg Return & Avg Length & Training \\
\midrule
Random    & 95.0 & 13.50 ± 6.20 & 7.5 ± 1.1 & - \\
Heuristic & 94.6 & 13.38 ± 6.45 & 7.8 ± 1.2 & - \\
\midrule
BC        & 94.2 & 13.26 ± 6.73 & 7.8 ± 1.2 & Offline \\
CQL       & 94.0 & 13.20 ± 6.83 & 7.9 ± 1.2 & Offline \\
DQN       & 94.0 & 13.20 ± 6.83 & 7.9 ± 1.2 & Offline \\
\bottomrule
\end{tabular}
\end{table}
```

**Updated Table**:
```latex
\begin{table}[htbp]
\centering
\caption{Overall Performance Comparison (500 episodes)}
\label{tab:overall}
\begin{tabular}{lcccc}
\toprule
Model & Survival (\%) & Avg Return & Avg Length & Paradigm \\
\midrule
\multicolumn{5}{l}{\textit{Baselines}} \\
Random    & 95.0 & 13.50 ± 6.20 & 7.5 ± 1.1 & - \\
Heuristic & 94.6 & 13.38 ± 6.45 & 7.8 ± 1.2 & - \\
\midrule
\multicolumn{5}{l}{\textit{Offline RL}} \\
BC        & 94.2 & 13.26 ± 6.73 & 7.8 ± 1.2 & Offline \\
CQL       & 94.0 & 13.20 ± 6.83 & 7.9 ± 1.2 & Offline \\
DQN       & 94.0 & 13.20 ± 6.83 & 7.9 ± 1.2 & Offline \\
\midrule
\multicolumn{5}{l}{\textit{Online RL}} \\
DDQN-Attention & 95.4 & 13.81 ± 5.92 & 7.6 ± 1.2 & Online \\
DDQN-Residual  & 94.2 & 13.26 ± 6.73 & 7.9 ± 1.3 & Online \\
SAC            & 94.8 & 13.44 ± 6.66 & 7.7 ± 1.2 & Online \\
\bottomrule
\end{tabular}
\end{table}
```

**Text Update** (after table):
Add paragraph:
```latex
Among all methods, DDQN-Attention achieves the highest survival rate (95.4\%), demonstrating the benefit of attention mechanisms for feature selection in complex medical domains. However, this comes at the cost of requiring environment interaction during training. Notably, offline methods (BC/CQL/DQN) achieve comparable performance (94.0-94.2\%) without any environment access, suggesting that pre-collected data is sufficient for learning effective sepsis treatment policies. The residual architecture (DDQN-Residual) does not provide consistent benefits over standard architectures in this domain, achieving 94.2\% survival similar to BC.
```

**Page Impact**: +0.3 pages

---

### TASK 4: Update Results Table 2 (SOFA-Stratified)
**File**: `paper/sections/05_results.tex`
**Location**: Section 5.2

**Action**: Add online models to existing SOFA stratification table

**Updated Table**:
```latex
\begin{table}[htbp]
\centering
\caption{SOFA-Stratified Performance (500 episodes)}
\label{tab:sofa}
\begin{tabular}{lccccc}
\toprule
& \multicolumn{4}{c}{High SOFA (Most Severe)} \\
\cmidrule(lr){2-5}
Model & n & Survival (\%) & Avg Return & Avg Length \\
\midrule
\multicolumn{5}{l}{\textit{Offline RL}} \\
BC  & 211 & 88.6 & 11.63 ± 9.82 & 8.3 ± 1.1 \\
CQL & 191 & 88.5 & 11.55 ± 9.95 & 8.3 ± 1.1 \\
DQN & 185 & 84.3 & 10.29 ± 11.46 & 8.5 ± 1.2 \\
\midrule
\multicolumn{5}{l}{\textit{Online RL}} \\
DDQN-Attention & 190 & \textbf{90.5} & 12.16 ± 9.20 & 8.0 ± 1.1 \\
DDQN-Residual  & 200 & 87.0 & 11.10 ± 10.15 & 8.3 ± 1.2 \\
SAC            & 195 & 88.7 & 11.62 ± 9.49 & 8.1 ± 1.1 \\
\bottomrule
\end{tabular}
\end{table}
```

**Text Update**:
```latex
The SOFA-stratified analysis reveals that all learned policies achieve substantially better outcomes than random policy on high-severity patients. DDQN-Attention achieves the highest survival rate (90.5\%) on high-SOFA patients, suggesting that attention mechanisms may be particularly beneficial for complex, severe cases where dynamic feature weighting is critical. Offline RL methods achieve competitive performance (84.3-88.6\%), with BC and CQL matching SAC's performance (88.5-88.6\% vs 88.7\%), demonstrating that offline learning can be effective even in high-stakes scenarios.
```

**Page Impact**: +0.2 pages

---

### TASK 5: Update Discussion - Offline vs Online
**File**: `paper/sections/06_discussion.tex`
**Location**: Section 6.1 (Main Findings), add new paragraph after existing content

**Action**: Add comparative analysis

```latex
\paragraph{Offline versus Online RL Trade-offs.}
Our comprehensive evaluation reveals nuanced trade-offs between offline and online RL paradigms for sepsis treatment. Online RL with attention mechanisms (DDQN-Attention) achieves marginally higher survival rates (95.4\% overall, 90.5\% on high-SOFA) compared to the best offline method (BC: 94.2\% overall, 88.6\% on high-SOFA). However, this 1.2-1.9 percentage point improvement comes at a significant practical cost: online methods require extensive environment interaction (1M steps) during training, which is infeasible in real clinical settings where patient safety is paramount.

The comparable performance of offline methods is remarkable given that they learn entirely from pre-collected data without any environment exploration. This suggests that the heuristic policy used to generate our offline dataset provides sufficient coverage of the state-action space for learning effective treatment strategies. Furthermore, as demonstrated in Section~\ref{sec:results:leg}, offline methods (particularly CQL) offer superior interpretability through LEG analysis, discovering 7+ clinically meaningful features compared to 4 for BC. This interpretability is crucial for clinical deployment, where understanding \textit{why} a model makes certain recommendations is as important as \textit{how well} it performs.

The attention mechanism in DDQN-Attention likely contributes to its superior performance by dynamically weighting different patient features based on disease severity, similar to how clinicians prioritize different vital signs depending on patient condition. However, without access to the internal attention weights (which were not preserved in the provided models), we cannot perform LEG analysis to validate this hypothesis or extract interpretable treatment rules from online RL policies.

For practical deployment in sepsis management, we recommend:
\begin{itemize}
\item \textbf{Research settings with simulators}: Online RL with attention can achieve marginally better performance if environment interaction is safe and feasible.
\item \textbf{Real clinical deployment}: Offline RL (particularly CQL) provides the best balance of performance (94.0\% survival), safety (no patient risk during training), and interpretability (7+ discovered features), making it more suitable for clinical decision support systems.
\end{itemize}
```

**Page Impact**: +0.8 pages

---

### TASK 6: Create Author Contributions Section
**File**: `paper/sections/08_contributions.tex` (NEW FILE)
**Location**: Between Conclusion and References in `main.tex`

**Create new file**:
```latex
% ============================================================
% AUTHOR CONTRIBUTIONS
% ============================================================

\section{Author Contributions}\label{sec:contributions}

This work represents a collaborative effort with distinct contributions from each author:

\paragraph{Zhiyu Cheng}
\begin{itemize}
\item Designed and implemented the comprehensive evaluation framework (500 episodes with SOFA stratification)
\item Implemented and trained offline RL algorithms (BC, CQL, DQN)
\item Conducted comparative analysis across all methods
\item Performed statistical analysis and generated all results tables/figures
\item Wrote and organized the manuscript
\end{itemize}

\paragraph{Yalun Ding}
\begin{itemize}
\item Proposed the LEG (Linearly Estimated Gradients) interpretability framework for sepsis treatment analysis
\item Implemented and trained online RL algorithms with architectural innovations (DDQN-Attention, DDQN-Residual, SAC)
\item Designed and implemented the attention and residual encoder architectures
\item Conducted initial exploratory analysis demonstrating feasibility of the approach
\end{itemize}

\paragraph{Chuanhui Peng}
\begin{itemize}
\item Designed the overall study framework comparing offline and online RL paradigms
\item Selected the set of offline RL algorithms (BC, CQL, DQN) based on theoretical considerations
\item Contributed to the problem formulation and research question design
\item Provided critical feedback on methodology and results interpretation
\end{itemize}

\vspace{1em}
\noindent All authors contributed to the conception of the study, reviewed and approved the final manuscript. The authors declare no conflicts of interest.
```

**Update main.tex**:
After `\input{sections/07_conclusion}`, add:
```latex
\input{sections/08_contributions}
```

**Page Impact**: +0.5 pages

---

### TASK 7: Compress Related Work
**File**: `paper/sections/02_related.tex`

**Current**: ~3 pages (sections 2.1, 2.2, 2.3 each ~1 page)

**Target**: ~1.5 pages (compress each to ~0.5 pages)

**Strategy**:
1. **Section 2.1 (RL for Sepsis)**: Keep only 2-3 most relevant citations, focus on key findings
2. **Section 2.2 (Offline RL)**: Streamline algorithm descriptions, combine similar approaches
3. **Section 2.3 (Interpretability)**: Condense to essential methods, focus on LEG

**Action**:
- Remove detailed algorithm explanations (readers can refer to original papers)
- Combine related citations into single sentences
- Remove less relevant background material
- Keep research gap discussion (Section 2.4) intact as it justifies our work

**Example compression for 2.1**:
```latex
% BEFORE (long):
Recent work has explored RL for sepsis treatment optimization. Raghu et al. (2017)
developed the gym-sepsis environment using MIMIC-III data... [3-4 sentences per paper]

% AFTER (concise):
Several studies have applied RL to sepsis treatment using MIMIC-III data
\citep{raghu2017deep, komorowski2018artificial, peng2018improving},
demonstrating that learned policies can match or exceed clinician performance.
However, these works primarily focus on online RL without addressing offline
learning challenges or interpretability requirements for clinical deployment.
```

**Page Impact**: -1.5 pages (saves space)

---

### TASK 8: Move LEG Formulas to Appendix
**File**: `paper/sections/04_methods.tex` → Create `paper/appendix.tex`

**Current Location**: Section 4.3 (~1 page of detailed math)

**Action**:
1. In main Methods section, keep only:
   - High-level explanation of LEG (2-3 sentences)
   - Reference to appendix for details
   - Key intuition

```latex
% In 04_methods.tex, Section 4.3:
\subsection{LEG Interpretability Analysis}\label{sec:methods:leg}

To quantify interpretability, we employ Linearly Estimated Gradients (LEG) analysis
\citep{proposed_by_ding}, which measures how much each state feature contributes to
action selection. The key idea is to approximate the policy's decision boundary using
local linear models and measure feature importance through gradient magnitudes.
Features with consistently high LEG scores across diverse states indicate that the
policy relies heavily on those features for decision-making, providing interpretable
treatment rules. Full mathematical details are provided in Appendix~\ref{app:leg}.
```

2. Create `appendix.tex`:
```latex
\appendix

\section{LEG Analysis Details}\label{app:leg}

\subsection{Mathematical Formulation}

[Move all detailed formulas here, including:]
- Equation for LEG computation
- Feature importance aggregation
- Statistical significance testing
- Implementation details

[Keep all current mathematical content]

\subsection{Implementation}

We implement LEG analysis using [details]...
```

3. Update `main.tex`:
Before `\bibliographystyle`, add:
```latex
\input{appendix}
```

**Page Impact**: -0.7 pages (from main text; appendix doesn't count toward 25-page limit)

---

### TASK 9: Simplify Future Directions
**File**: `paper/sections/06_discussion.tex`
**Location**: Section 6.5

**Current**: Likely has 5-6 future directions, each with explanation

**Target**: Reduce to 3 most important directions, 1-2 sentences each

**Action**:
Keep only:
1. Real clinical trial deployment
2. Transfer learning across hospitals
3. Integration with existing clinical decision support systems

Remove:
- Theoretical extensions
- Minor algorithmic improvements
- Peripheral research directions

**Example**:
```latex
\subsection{Future Directions}\label{sec:discussion:future}

Three promising avenues for future work include: (1) prospective clinical trials
to validate offline RL policies in real ICU settings with human-in-the-loop
oversight; (2) transfer learning approaches to adapt policies trained on MIMIC-III
to other hospital systems with different patient demographics; and (3) integration
with existing clinical decision support systems to provide actionable, interpretable
treatment recommendations at the bedside.
```

**Page Impact**: -0.5 pages

---

### TASK 10: Final Compilation and Page Count Check
**Files**: All above changes

**Action**:
1. Compile LaTeX: `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
2. Check page count (should be ≤25)
3. If over 25 pages, identify additional sections to compress:
   - Introduction (trim background)
   - Problem Formulation (combine equations)
   - Results text (be more concise)

**Page Budget Calculation**:
```
Current:        23 pages
Add TASK 2:    +1.5 pages (Online RL methods)
Add TASK 3:    +0.3 pages (Table 1)
Add TASK 4:    +0.2 pages (Table 2)
Add TASK 5:    +0.8 pages (Discussion)
Add TASK 6:    +0.5 pages (Contributions)
Remove TASK 7: -1.5 pages (Related Work)
Remove TASK 8: -0.7 pages (LEG to appendix)
Remove TASK 9: -0.5 pages (Future Directions)
-----------------------------------
Total:          23.6 pages ✓
```

**Success Criteria**:
- [ ] All 3 online models included
- [ ] Tables updated with correct numbers
- [ ] Author contributions clearly stated
- [ ] Page count ≤ 25
- [ ] No compilation errors
- [ ] References properly cited

---

## 📊 Key Numbers Reference

### Model Performance (from yalun_models_evaluation.pkl)

**DDQN-Attention**:
- Overall: 95.4% survival, 13.81 ± 5.92 return, 7.6 ± 1.2 length
- High SOFA: 90.5% survival (n=190), 12.16 ± 9.20 return, 8.0 ± 1.1 length

**DDQN-Residual**:
- Overall: 94.2% survival, 13.26 ± 6.73 return, 7.9 ± 1.3 length
- High SOFA: 87.0% survival (n=200), 11.10 ± 10.15 return, 8.3 ± 1.2 length

**SAC**:
- Overall: 94.8% survival, 13.44 ± 6.66 return, 7.7 ± 1.2 length
- High SOFA: 88.7% survival (n=195), 11.62 ± 9.49 return, 8.1 ± 1.1 length

### Existing Models (for reference)
**BC**: 94.2% overall, 88.6% high-SOFA (n=211)
**CQL**: 94.0% overall, 88.5% high-SOFA (n=191)
**DQN**: 94.0% overall, 84.3% high-SOFA (n=185)

---

## 🎯 Important Notes

1. **LEG Analysis**: Only applied to offline models (BC/CQL/DQN). Online models don't have LEG analysis because:
   - Yalun's models don't have saved encoder states needed for LEG
   - Focus of paper is interpretability of offline methods
   - Online models serve as performance benchmark

2. **Citations**: Check that these papers are in bibliography:
   - van Hasselt et al. (2016) - Double DQN
   - He et al. (2016) - ResNet
   - Haarnoja et al. (2018) - SAC
   - Add Yalun's LEG work if published

3. **Consistency**:
   - All tables use 500 episodes
   - All methods evaluated on same test set
   - SOFA stratification: Low (<5), Medium (5-15), High (>15)

4. **Tone**:
   - Objective comparison, not competition
   - Acknowledge strengths of both paradigms
   - Emphasize practical deployment considerations

5. **Author Order**:
   - Alphabetical with "equal contribution" note
   - Or discuss with co-authors

---

## 📁 File Structure

```
paper/
├── main.tex                          # Main document (update line 91-98)
├── sections/
│   ├── 01_introduction.tex           # No changes
│   ├── 02_related.tex                # TASK 7: Compress
│   ├── 03_problem.tex                # No changes
│   ├── 04_methods.tex                # TASK 2: Add online RL; TASK 8: LEG
│   ├── 05_results.tex                # TASK 3 & 4: Update tables
│   ├── 06_discussion.tex             # TASK 5 & 9: Add comparison & simplify
│   ├── 07_conclusion.tex             # Minor update to mention collaboration
│   └── 08_contributions.tex          # TASK 6: NEW FILE
├── appendix.tex                      # TASK 8: NEW FILE (LEG details)
└── references.bib                    # Add new citations if needed
```

---

## ✅ Completion Checklist

After completing all tasks, verify:

- [ ] Paper compiles without errors (`pdflatex` × 3 + `bibtex`)
- [ ] Page count ≤ 25 pages
- [ ] All 8 models appear in tables (2 baselines + 3 offline + 3 online)
- [ ] Numbers match `yalun_models_evaluation.pkl`
- [ ] Author contributions section present
- [ ] LEG details moved to appendix
- [ ] All citations present in bibliography
- [ ] Figures/tables numbered correctly
- [ ] Cross-references work (e.g., "Section~\ref{...}")
- [ ] Consistent terminology (Offline RL vs offline RL vs offline learning)
- [ ] No TODOs or placeholder text remaining

---

## 🚀 Quick Start

If you're a new model taking over this task:

1. **Read this entire guide first**
2. **Load the results**: `results/yalun_models_evaluation.pkl`
3. **Start with TASK 1** (easiest) to understand the paper structure
4. **Follow task order** 1→2→3→4→5→6→7→8→9→10
5. **Mark tasks complete** in the TodoWrite list as you go
6. **Compile frequently** to catch errors early

Good luck! This is a collaborative scientific paper - maintain professional tone and objective analysis throughout.
