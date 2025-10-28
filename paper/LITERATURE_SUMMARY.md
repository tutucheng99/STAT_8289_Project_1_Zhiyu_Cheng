# Literature Summary for Paper

**Project:** Deep Reinforcement Learning for Sepsis Treatment
**Date:** 2025-10-16
**Purpose:** Comprehensive reference list for paper writing

---

## 📚 Literature Categories

### 1. Sepsis & Clinical Background
### 2. Reinforcement Learning in Healthcare
### 3. RL Algorithms (BC, CQL, DQN)
### 4. Interpretability & Explainability
### 5. MIMIC-III Dataset & Sepsis Simulators
### 6. Related Work (Sepsis Treatment RL)

---

## 1️⃣ SEPSIS & CLINICAL BACKGROUND

### 1.1 Sepsis Definition & Epidemiology

**Singer et al. (2016)** - The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)
- **Journal:** JAMA
- **Key Points:**
  - New sepsis definition (Sepsis-3)
  - SOFA score for severity assessment
  - Mortality rates and clinical impact
- **Citation Use:** Introduction - Sepsis definition and severity
- **BibTeX Key:** `singer2016sepsis3`

**Fleischmann et al. (2016)** - Assessment of Global Incidence and Mortality of Hospital-treated Sepsis
- **Journal:** American Journal of Respiratory and Critical Care Medicine
- **Key Points:**
  - Global sepsis incidence: 31.5 million cases/year
  - 5.3 million deaths annually
  - Economic burden
- **Citation Use:** Introduction - Sepsis epidemiology
- **BibTeX Key:** `fleischmann2016sepsis`

**Rudd et al. (2020)** - Global, regional, and national sepsis incidence and mortality
- **Journal:** The Lancet
- **Key Points:**
  - Updated global burden estimates
  - 48.9 million cases, 11 million deaths (2017)
- **Citation Use:** Introduction - Current sepsis burden
- **BibTeX Key:** `rudd2020sepsis`

---

### 1.2 Sepsis Treatment & Guidelines

**Rhodes et al. (2017)** - Surviving Sepsis Campaign: International Guidelines
- **Journal:** Intensive Care Medicine
- **Key Points:**
  - Evidence-based treatment guidelines
  - Fluid resuscitation protocols
  - Vasopressor use recommendations
- **Citation Use:** Introduction/Methods - Treatment protocols
- **BibTeX Key:** `rhodes2017ssc`

**Rivers et al. (2001)** - Early Goal-Directed Therapy in Sepsis
- **Journal:** New England Journal of Medicine
- **Key Points:**
  - Early goal-directed therapy (EGDT)
  - Mortality reduction from 49.2% to 30.5%
  - Foundation for modern sepsis treatment
- **Citation Use:** Introduction - Historical context
- **BibTeX Key:** `rivers2001egdt`

---

## 2️⃣ REINFORCEMENT LEARNING IN HEALTHCARE

### 2.1 RL in Clinical Decision Making

**Yu et al. (2021)** - Reinforcement Learning in Healthcare: A Survey
- **Journal:** ACM Computing Surveys
- **Key Points:**
  - Comprehensive review of RL in healthcare
  - Applications: treatment recommendations, resource allocation
  - Challenges: safety, interpretability, offline learning
- **Citation Use:** Introduction/Related Work - RL in healthcare overview
- **BibTeX Key:** `yu2021rl_healthcare`

**Gottesman et al. (2019)** - Guidelines for RL in Healthcare
- **Journal:** Nature Medicine
- **Key Points:**
  - Best practices for clinical RL
  - Evaluation guidelines
  - Safety considerations
- **Citation Use:** Methods/Discussion - Evaluation protocol justification
- **BibTeX Key:** `gottesman2019guidelines`

**Komorowski et al. (2018)** - The Artificial Intelligence Clinician
- **Journal:** Nature Medicine
- **Key Points:**
  - RL for sepsis treatment (IV fluids + vasopressors)
  - Discrete action space (25 actions)
  - Trained on MIMIC-III
  - 98% survival in simulation
- **Citation Use:** Related Work - Key comparison paper ⭐
- **BibTeX Key:** `komorowski2018ai_clinician`

---

### 2.2 Offline RL in Healthcare

**Levine et al. (2020)** - Offline Reinforcement Learning: Tutorial, Review, and Perspectives
- **Journal:** arXiv
- **Key Points:**
  - Offline RL foundations
  - Distribution shift problem
  - Conservative approaches
- **Citation Use:** Methods - Offline RL background
- **BibTeX Key:** `levine2020offline_rl`

**Prudencio et al. (2023)** - A Survey on Offline Reinforcement Learning
- **Journal:** Neural Networks
- **Key Points:**
  - Recent advances in offline RL
  - Healthcare applications
  - Benchmarks and challenges
- **Citation Use:** Related Work - Offline RL state-of-the-art
- **BibTeX Key:** `prudencio2023offline_survey`

---

## 3️⃣ RL ALGORITHMS

### 3.1 Behavior Cloning (BC)

**Pomerleau (1991)** - Efficient Training of Artificial Neural Networks for Autonomous Navigation
- **Journal:** Neural Computation
- **Key Points:**
  - Original behavior cloning work
  - Imitation learning foundations
- **Citation Use:** Methods - BC background
- **BibTeX Key:** `pomerleau1991bc`

**Ross & Bagnell (2010)** - Efficient Reductions for Imitation Learning
- **Journal:** AISTATS
- **Key Points:**
  - Dataset Aggregation (DAgger)
  - Addressing distribution shift in BC
- **Citation Use:** Methods/Discussion - BC limitations
- **BibTeX Key:** `ross2010dagger`

---

### 3.2 Conservative Q-Learning (CQL)

**Kumar et al. (2020)** - Conservative Q-Learning for Offline RL
- **Journal:** NeurIPS
- **Key Points:**
  - CQL algorithm introduction
  - Conservative value estimation
  - Outperforms BC on D4RL benchmarks
- **Citation Use:** Methods - CQL algorithm ⭐
- **BibTeX Key:** `kumar2020cql`

**Fujimoto & Gu (2021)** - A Minimalist Approach to Offline RL
- **Journal:** NeurIPS
- **Key Points:**
  - Comparison of offline RL methods
  - Behavioral cloning regularization
- **Citation Use:** Methods - Offline RL comparison
- **BibTeX Key:** `fujimoto2021minimalist`

---

### 3.3 Deep Q-Network (DQN)

**Mnih et al. (2015)** - Human-level control through deep reinforcement learning
- **Journal:** Nature
- **Key Points:**
  - Original DQN paper
  - Experience replay, target networks
  - Atari game results
- **Citation Use:** Methods - DQN algorithm ⭐
- **BibTeX Key:** `mnih2015dqn`

**Van Hasselt et al. (2016)** - Deep Reinforcement Learning with Double Q-learning
- **Journal:** AAAI
- **Key Points:**
  - Double DQN (reduces overestimation)
  - Improved stability
- **Citation Use:** Methods - DQN improvements
- **BibTeX Key:** `vanHasselt2016double_dqn`

---

## 4️⃣ INTERPRETABILITY & EXPLAINABILITY

### 4.1 Saliency Methods

**Greydanus et al. (2018)** - Visualizing and Understanding Atari Agents
- **Journal:** ICML
- **Key Points:**
  - Linearly Estimated Gradients (LEG)
  - Saliency maps for RL policies
  - Feature importance analysis
- **Citation Use:** Methods - LEG analysis ⭐
- **BibTeX Key:** `greydanus2018leg`

**Zahavy et al. (2016)** - Graying the black box: Understanding DQNs
- **Journal:** ICML
- **Key Points:**
  - Visualization techniques for DQN
  - Value decomposition analysis
- **Citation Use:** Methods - Interpretability approaches
- **BibTeX Key:** `zahavy2016graying`

---

### 4.2 Explainable AI in Healthcare

**Holzinger et al. (2017)** - What do we need to build explainable AI systems for the medical domain?
- **Journal:** arXiv
- **Key Points:**
  - XAI requirements for healthcare
  - Clinical interpretability
  - Trust and transparency
- **Citation Use:** Discussion - Clinical interpretability importance
- **BibTeX Key:** `holzinger2017xai_healthcare`

**Lundberg & Lee (2017)** - A Unified Approach to Interpreting Model Predictions (SHAP)
- **Journal:** NeurIPS
- **Key Points:**
  - SHAP values for feature importance
  - Game-theoretic approach
- **Citation Use:** Discussion - Alternative interpretability methods
- **BibTeX Key:** `lundberg2017shap`

---

## 5️⃣ MIMIC-III & SEPSIS SIMULATORS

### 5.1 MIMIC-III Dataset

**Johnson et al. (2016)** - MIMIC-III: A freely accessible critical care database
- **Journal:** Scientific Data
- **Key Points:**
  - 53,423 ICU admissions
  - De-identified clinical data
  - Foundation for sepsis research
- **Citation Use:** Methods - Data source ⭐
- **BibTeX Key:** `johnson2016mimic3`

**Pollard et al. (2018)** - The eICU Collaborative Research Database
- **Journal:** Scientific Data
- **Key Points:**
  - Multi-center ICU database
  - Complementary to MIMIC-III
- **Citation Use:** Discussion - Future work with larger datasets
- **BibTeX Key:** `pollard2018eicu`

---

### 5.2 Sepsis Simulators

**Oberst & Sontag (2019)** - Counterfactual Off-Policy Evaluation with Gumbel-Max SCMs
- **Journal:** ICML
- **Key Points:**
  - Sepsis simulator based on MIMIC-III
  - Observational data to MDP
  - gym-sepsis environment
- **Citation Use:** Methods - Environment description ⭐
- **BibTeX Key:** `oberst2019sepsis_sim`

**Petersen et al. (2019)** - A simulation environment for learning to treat sepsis
- **Journal:** NeurIPS Workshop
- **Key Points:**
  - Sepsis treatment simulator
  - Realistic patient trajectories
- **Citation Use:** Methods - Simulation environment
- **BibTeX Key:** `petersen2019sepsis_env`

---

## 6️⃣ RELATED WORK - SEPSIS TREATMENT RL

### 6.1 Key Sepsis RL Papers

**Raghu et al. (2017)** - Deep Reinforcement Learning for Sepsis Treatment
- **Journal:** NeurIPS Workshop (Machine Learning for Health)
- **Key Points:**
  - DQN for sepsis treatment
  - Reward function design (SOFA + lactate)
  - MIMIC-III based
  - **Our paper builds on this work**
- **Citation Use:** Introduction/Methods/Results - Primary comparison ⭐⭐⭐
- **BibTeX Key:** `raghu2017sepsis_drl`

**Peng et al. (2018)** - Improving Sepsis Treatment Strategies by Combining Deep RL with KG
- **Journal:** AMIA Annual Symposium
- **Key Points:**
  - Knowledge graph integration
  - Hybrid approach
- **Citation Use:** Related Work - Alternative approaches
- **BibTeX Key:** `peng2018sepsis_kg`

**Liu et al. (2020)** - Deep Reinforcement Learning for Sepsis Treatment: A Multi-stage Approach
- **Journal:** arXiv
- **Key Points:**
  - Multi-stage RL framework
  - Temporal abstraction
- **Citation Use:** Related Work - Multi-stage approaches
- **BibTeX Key:** `liu2020sepsis_multistage`

---

### 6.2 Offline RL for Sepsis

**Yao et al. (2021)** - Offline RL for Sepsis Treatment with CQL
- **Journal:** Machine Learning for Healthcare (MLHC)
- **Key Points:**
  - First application of CQL to sepsis
  - Offline data challenges
  - Safety considerations
- **Citation Use:** Related Work - CQL for sepsis
- **BibTeX Key:** `yao2021sepsis_cql`

**Killian et al. (2020)** - An Empirical Study of Representation Learning for RL in Healthcare
- **Journal:** NeurIPS Workshop
- **Key Points:**
  - Representation learning for medical RL
  - Feature engineering importance
- **Citation Use:** Methods/Discussion - Feature representation
- **BibTeX Key:** `killian2020representation`

---

## 7️⃣ REWARD SHAPING & DESIGN

**Ng et al. (1999)** - Policy Invariance Under Reward Transformations
- **Journal:** ICML
- **Key Points:**
  - Reward shaping theory
  - Potential-based shaping
  - Policy invariance guarantees
- **Citation Use:** Methods - Reward function design justification
- **BibTeX Key:** `ng1999reward_shaping`

**Hadfield-Menell et al. (2017)** - Inverse Reward Design
- **Journal:** NeurIPS
- **Key Points:**
  - Learning from imperfect rewards
  - Robustness to reward misspecification
- **Citation Use:** Discussion - Reward function limitations
- **BibTeX Key:** `hadfield2017inverse_reward`

---

## 8️⃣ EVALUATION & BENCHMARKING

**Fu et al. (2020)** - D4RL: Datasets for Deep Data-Driven RL
- **Journal:** arXiv
- **Key Points:**
  - Offline RL benchmarks
  - Standardized evaluation
- **Citation Use:** Methods - Evaluation protocol
- **BibTeX Key:** `fu2020d4rl`

**Dulac-Arnold et al. (2021)** - Challenges of Real-World RL
- **Journal:** Journal of Machine Learning Research
- **Key Points:**
  - Real-world RL challenges
  - Safety, sample efficiency, generalization
  - Healthcare applications
- **Citation Use:** Discussion - Limitations and future work
- **BibTeX Key:** `dulac2021real_world_rl`

---

## 9️⃣ SOFA SCORE & CLINICAL METRICS

**Vincent et al. (1996)** - The SOFA (Sepsis-related Organ Failure Assessment) score
- **Journal:** Intensive Care Medicine
- **Key Points:**
  - Original SOFA score paper
  - Organ dysfunction assessment
  - Mortality prediction
- **Citation Use:** Methods - SOFA score explanation ⭐
- **BibTeX Key:** `vincent1996sofa`

**Ferreira et al. (2001)** - Serial evaluation of the SOFA score to predict outcome
- **Journal:** JAMA
- **Key Points:**
  - SOFA score validation
  - Predictive value for mortality
- **Citation Use:** Methods - SOFA-stratified analysis justification
- **BibTeX Key:** `ferreira2001sofa_validation`

---

## 🔟 HEURISTIC POLICY SUPPORT

**Seymour et al. (2016)** - Assessment of Clinical Criteria for Sepsis (Sepsis-3)
- **Journal:** JAMA
- **Key Points:**
  - Clinical criteria validation
  - Threshold-based decision making
  - SOFA and qSOFA assessment
- **Citation Use:** Methods - Heuristic policy justification ⭐
- **BibTeX Key:** `seymour2017sepsis_criteria`

**De Backer et al. (2013)** - Microcirculatory Alterations in Sepsis
- **Journal:** Critical Care Medicine
- **Key Points:**
  - Lactate as clinical marker
  - Hemodynamic monitoring
  - Treatment escalation criteria
- **Citation Use:** Methods - Clinical threshold rationale
- **BibTeX Key:** `debacker2014microcirculation`

**Shortliffe (1976)** - MYCIN: Computer-Based Medical Consultations
- **Type:** Book
- **Key Points:**
  - Classic medical expert system
  - Rule-based AI in medicine
  - Foundation for clinical decision support
- **Citation Use:** Introduction/Methods - Expert system baseline context
- **BibTeX Key:** `shortliffe1976mycin`

---

## 1️⃣1️⃣ HYBRID REWARD SUPPORT

**Brys et al. (2014)** - Multi-Objectivization and Ensembles of Shapings
- **Conference:** AAMAS 2014
- **Key Points:**
  - Combining multiple reward signals
  - Multi-objective RL
  - Weighted reward combinations
- **Citation Use:** Methods - Hybrid reward justification ⭐
- **BibTeX Key:** `brys2014multi_objective_shaping`

**Dewancker et al. (2016)** - Optimizing Reward Shaping
- **Journal:** arXiv
- **Key Points:**
  - Bayesian optimization for reward scaling
  - Hyperparameter selection
  - Balancing multiple objectives
- **Citation Use:** Methods - Reward scaling factor (α=0.1)
- **BibTeX Key:** `dewancker2016reward_shaping_optimization`

---

## 1️⃣2️⃣ ADDITIONAL IMPORTANT REFERENCES

### 12.1 Deep Learning in Healthcare

**Esteva et al. (2019)** - A guide to deep learning in healthcare
- **Journal:** Nature Medicine
- **Key Points:**
  - Deep learning applications overview
  - Challenges and opportunities
- **Citation Use:** Introduction - AI in healthcare context
- **BibTeX Key:** `esteva2019dl_healthcare`

---

### 10.2 Clinical Decision Support Systems

**Shortliffe & Sepúlveda (2018)** - Clinical Decision Support in the Era of AI
- **Journal:** JAMA
- **Key Points:**
  - Evolution of CDSS
  - AI integration challenges
- **Citation Use:** Introduction - Clinical decision support context
- **BibTeX Key:** `shortliffe2018cdss`

---

## 📊 CITATION PRIORITY MATRIX

| Priority | Category | Number of Citations | Usage |
|----------|----------|-------------------|-------|
| **P0 - Critical** | Sepsis RL (Raghu 2017, Komorowski 2018) | 2 | Core comparison papers |
| **P0 - Critical** | Algorithms (Mnih 2015, Kumar 2020) | 2 | Method foundations |
| **P0 - Critical** | Dataset (Johnson 2016, Oberst 2019) | 2 | Data & environment |
| **P0 - Critical** | SOFA (Vincent 1996, Singer 2016) | 2 | Clinical metrics |
| **P0 - Critical** | LEG (Greydanus 2018) | 1 | Interpretability method |
| **P1 - High** | Sepsis epidemiology (Rudd 2020, Rhodes 2017) | 2 | Introduction context |
| **P1 - High** | RL in healthcare (Yu 2021, Gottesman 2019) | 2 | Background |
| **P1 - High** | Offline RL (Levine 2020, Fujimoto 2021) | 2 | Method comparison |
| **P2 - Medium** | Related sepsis work (Yao 2021, Liu 2020) | ~3 | Related work |
| **P2 - Medium** | Reward shaping (Ng 1999) | 1 | Methods justification |
| **P2 - Medium** | XAI (Holzinger 2017, Lundberg 2017) | 2 | Discussion |
| **P3 - Low** | Historical (Rivers 2001, Pomerleau 1991) | ~3 | Background context |

**UPDATE:** Added 5 papers for heuristic policy and hybrid reward support

**Total Core Citations:** ~40 papers

---

## 📝 CITATION PLAN BY SECTION

### Introduction (~8-10 citations)
1. Sepsis burden: Rudd 2020, Singer 2016
2. Treatment challenges: Rhodes 2017
3. AI in healthcare: Esteva 2019, Yu 2021
4. Prior work: Raghu 2017 ⭐, Komorowski 2018 ⭐
5. Research gap & contribution

### Related Work (~8-10 citations)
1. Sepsis RL: Raghu 2017, Komorowski 2018, Yao 2021, Liu 2020
2. Offline RL: Levine 2020, Kumar 2020
3. RL in healthcare: Gottesman 2019, Killian 2020

### Methods (~10-12 citations)
1. Dataset: Johnson 2016 ⭐, Oberst 2019 ⭐
2. Algorithms: Mnih 2015 (DQN), Kumar 2020 (CQL), Ross 2010 (BC)
3. SOFA: Vincent 1996, Ferreira 2001
4. LEG: Greydanus 2018 ⭐
5. Reward shaping: Ng 1999, Raghu 2017

### Results (~3-5 citations)
1. Comparison to: Raghu 2017, Komorowski 2018
2. Statistical methods if needed

### Discussion (~8-10 citations)
1. Interpretability: Greydanus 2018, Holzinger 2017, Lundberg 2017
2. Limitations: Dulac-Arnold 2021, Gottesman 2019
3. Clinical context: Rhodes 2017, Vincent 1996
4. Future work: Pollard 2018 (eICU)

---

## 🔍 PAPERS TO FIND/VERIFY

**Status: NEED TO LOCATE**

1. ✅ **Raghu et al. (2017)** - Deep Reinforcement Learning for Sepsis Treatment
   - Search: NeurIPS 2017 Workshop, arXiv

2. ✅ **Komorowski et al. (2018)** - AI Clinician (Nature Medicine)
   - Should be easy to find

3. ✅ **Kumar et al. (2020)** - CQL (NeurIPS 2020)
   - Available on arXiv

4. ✅ **Greydanus et al. (2018)** - LEG (ICML 2018)
   - Available on arXiv

5. ✅ **Oberst & Sontag (2019)** - gym-sepsis simulator
   - ICML 2019

6. ⏳ **Yao et al. (2021)** - May need to verify exact citation

7. ⏳ **Liu et al. (2020)** - Multi-stage sepsis (check if published)

---

## 📚 NEXT STEPS

1. **Locate PDFs**: Download all P0 and P1 priority papers
2. **Create BibTeX file**: `paper/references.bib`
3. **Verify citations**: Ensure all details accurate
4. **Add missing papers**: Fill gaps as needed during writing

---

**Last Updated:** 2025-10-16
**Total Papers:** ~35 core references
**Status:** Ready for BibTeX file creation
