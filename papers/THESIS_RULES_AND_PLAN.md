# Master Thesis — Rules, Plan & Working Guidelines

## Author: Utkarsh Sawant
## Institution: SRH Hochschule Heidelberg
## Topic: Data-Driven Approaches for Bike Sharing Demand Prediction
## Date Created: February 16, 2026

---

# PART A: DOCUMENT MANAGEMENT RULES

## A.1 Primary Document
- The PRIMARY working document is: `papers/Utkarsh_Master_Thesis.docx` (and its PDF export `papers/Utkarsh_Master_Thesis.pdf`)
- This is the ONLY document that matters for submission
- NEVER edit this file programmatically — all edits are done by the user manually in Word
- When we say "the document" or "the thesis" — it always means this file

## A.2 Secondary Reference File
- `FINAL_THESIS_DOCUMENT.md` exists as a secondary reference only
- It contains raw content, model details, and metrics for reference
- Do NOT treat it as the working document
- Do NOT edit it unless explicitly asked

## A.3 Content Workflow
1. Claude drafts content in chat (flowing academic prose)
2. User reviews, humanizes, and adjusts the text
3. User manually inserts approved content into Utkarsh_Master_Thesis.docx
4. Claude NEVER edits either the .docx or the .md unless explicitly commanded

## A.4 Reference Papers
- `papers/` folder contains citation source PDFs — use them for accurate references
- `MA_Thesis_AMBIKA_NAIR.pdf` (root folder) — style reference from professor's previous student
- `papers/thesis writinig guidelines.pdf` — professor's recorded lecture on thesis structure
- Web search permitted for finding additional citations

---

# PART B: WRITING STYLE RULES (Professor's Strict Requirements)

## B.1 Anti-AI Detection — CRITICAL
The professor explicitly checks for AI-generated content. These rules are non-negotiable:

1. **NO bullet points** — scientific papers and theses NEVER have bullet points. Everything must be flowing prose paragraphs
2. **NO bold text in body** — no italic, no bold anywhere in the body text. Only chapter titles, section headings, and figure/table captions may be formatted
3. **NO AI-obvious structure** — NEVER write "Advantages of X / Disadvantages of X / Summary of X" for each topic. This pattern is an immediate red flag
4. **Figures every 1-2 pages** — a thesis with only text and no figures is a clear AI sign. Ch 1-2: figures from published sources with citations. Ch 4-6: primarily author-generated (code screenshots, architecture diagrams, result plots)
5. **NO AI-generated images without citations** — Ch 1-2 figures must come from published sources with citations. Ch 4-6 figures are author-generated and labelled "[author]"
6. **Verify ALL references** — AI hallucinates references. Every reference must be verified for correct author, title, year, and venue. Non-existent references are undeniable proof of AI
7. **NO second person** — never use "you" or "your" in the thesis text
8. **Natural transitions** — do not use mechanical/repetitive transition sentences between sections

## B.2 Academic Tone — MANDATORY
Every sentence in the thesis must pass an academic tone check. The following are explicitly forbidden:

1. **NO casual or conversational language** — avoid contractions (it's, don't, can't), chatty asides, blog-like phrasing. Write formally at all times
2. **NO slang, idioms, or colloquial expressions** — phrases like "way over the top", "pretty common", "a touch deceiving", "huge gap" are not acceptable. Use precise academic alternatives
3. **NO emotional or dramatic wording** — avoid intensifiers like "absolutely", "wildly", "miserably", "oppressive". State facts neutrally
4. **NO overgeneralisations or sweeping claims** — every claim must be scoped and, where possible, supported by a citation
5. **NO rhetorical questions** — never pose a question to the reader in body text
6. **NO vague or imprecise statements** — replace "a lot", "some", "kind of", "sort of" with specific quantities or precise language
7. **NO strong personal opinions without evidence** — assertions must be grounded in data or literature
8. **NO first person singular** — avoid "I" and "my". Use "this study", "this thesis", "the present work", or the authorial "we" sparingly
9. **NO anthropomorphising models** — do not say a model "sees", "knows", "learns", "its job is to". Describe what the model computes or produces
10. **Consistent register** — do not mix formal and informal language within or across paragraphs. Maintain a uniform academic voice throughout

Quick self-check: if a sentence could appear in a blog post, a Reddit comment, or a product description, it does not belong in the thesis.

## B.3 Academic Prose Style
- Follow Ambika's thesis style — flowing paragraphs, natural academic tone
- No numbered phases/steps in body text — use natural transitions
- Keep introductory text brief — save detail for later chapters
- Comparison done ONCE at the end (section 2.5), not per technology
- Vary sentence structure and paragraph openings — avoid repetitive patterns
- Quality over quantity — concise, well-argued paragraphs are preferred over lengthy filler

## B.4 Figures, Tables & Formulas
- Insert a figure or table every 1-2 pages throughout the thesis
- Every figure needs: a caption, a citation/source, and a reference in the text
- Use figure placeholders when drafting: describe what the figure should show, placed inline where referenced in the prose
- For code screenshots: specify the exact file path and line numbers so user can take the screenshot
- Tables must have proper captions and be referenced in the text
- Figures reduce text burden and make the thesis look less AI-generated

### Numbering Convention (section-based):
- Figures: Figure 4.3.1, Figure 4.3.2 (chapter.section.sequence)
- Tables: Table 4.3.1, Table 4.3.2
- Formulas: Formula 4.3.1, Formula 4.3.2

### Formula Placeholders:
- User will insert formulas as images in the .docx — Claude drafts placeholder marks in text
- Use placeholder format: [Formula X.Y.Z: brief description of what the formula shows]
- Every formula must be referenced naturally in the prose before the placeholder appears
- Example: "The cyclical encoding is defined in Formula 4.3.1." followed by [Formula 4.3.1: Sine-cosine cyclical encoding for temporal variables]

### Formula Placement Rules:
- **Chapter 2 (State of the Art):** Theoretical/foundational formulas for each model architecture — LSTM gates, GRU gates, TCN dilated convolution, attention mechanism, XGBoost objective. These define how each architecture works mathematically.
- **Chapter 3 (Methodology):** NO formulas — blueprint only
- **Chapter 4 (Implementation):** Applied/configuration formulas only — loss functions (MSE), feature engineering equations (cyclical encoding, correlation), CBAM attention (our novel contribution), mixup augmentation, receptive field calculation. These are formulas tied to our specific implementation choices.
- **Rule of thumb:** If the formula explains "what the model IS" → Chapter 2. If the formula explains "what WE configured or built" → Chapter 4.

### Chapter 2 Formula Backlog (TO BE ADDED after Chapter 4 is complete):
- Section 2.1.3 (LSTM): forget gate, input gate, candidate state, cell state update, output gate, hidden state
- Section 2.1.4 (GRU): update gate, reset gate, candidate state, hidden state
- Section 2.1.5 (TCN): dilated causal convolution equation
- Section 2.1.6 (Attention): scaled dot-product attention, multi-head attention
- Section 2.1.7 (XGBoost): objective function with regularisation term

## B.5 Citation Style
- IEEE numbered bracket format: [1], [2], [3]
- References [1]-[5] are LOCKED — they match the original .docx and must never change
- New references assigned from [50]+ (references [6]-[49] already assigned in .docx)
- Currently assigned reference highlights:
  - [6] Wirth & Hipp 2000 (CRISP-DM)
  - [7] Sathishkumar & Cho 2020 (Seoul bike rule-based)
  - [8] Hyndman & Athanasopoulos 2018 (forecasting textbook)
  - [9] Lim & Zohren 2021 (time-series forecasting with deep learning survey)
  - [10] Benidis et al. 2023 (deep learning for time series survey)
  - [11] Box & Jenkins 1976 (time series analysis textbook)
  - [12] Chen & Guestrin 2016 (XGBoost)
  - [13] Hochreiter & Schmidhuber 1997 (LSTM)
  - [14] Bai et al. 2018 (TCN)
  - [15] Vaswani et al. 2017 (Attention)
  - [16] Cho et al. 2014 (GRU)
  - [17] Woo et al. 2018 (CBAM)
  - [18] Goodfellow et al. 2016 (Deep Learning textbook)
  - [19] LeCun et al. 2015 (Deep learning, Nature)
  - [20] Rumelhart et al. 1986 (backpropagation)
  - [21] Bengio et al. 1994 (vanishing gradient problem)
  - [22] Fraunhofer IST (CRISP-DM diagram image)
  - [23] Distill.pub (unrolled RNN architecture image)
  - [24] LinkedIn (LSTM Network image)
  - [25] Min & Jung 2021 (Seoul bike prediction comparison study)
  - [26] Slideserve (supervised ML pipeline image)
  - [27] Chung et al. 2014 (GRU empirical evaluation)
  - [28] Fu et al. 2016 (LSTM and GRU for traffic flow prediction)
  - [29] He et al. 2016 (ResNet / deep residual learning)
  - [30] Hewage et al. 2020 (TCN for weather forecasting)
  - [31] Ma et al. 2015 (LSTM for traffic speed prediction)
  - [32] Xu et al. 2021 (ITS sensing meets edge computing)
  - [33] Lin et al. 2018 (station-level bike-sharing demand prediction)
  - [34] Bahdanau et al. 2015 (attention — Neural Machine Translation)
  - [35] Luong et al. 2015 (attention — Effective Approaches)
  - [36] Breiman 1996 (bagging predictors)
  - [37] Freund & Schapire 1997 (boosting)
  - [38] Friedman 2001 (gradient boosting machine)
  - [39] Ergul Aydin et al. 2023 (bike-sharing demand with gradient boosting)
  - [40] Aghe 2023 (Seoul bike sharing stacking regressor)
  - [41] Li et al. 2023 (irregular CNN for bike sharing demand)
  - [42] Zhou et al. 2022 (hybrid TCN-GRU for bike-sharing demand)
  - [43] ResearchGate (GRU cell architecture image)
  - [44] arXiv / WaveNet (stacked recurrent network image)
  - [45] Medium (CBAM architecture image)
  - [46] Google Gemini generated image (decision tree vs bagging vs gradient boosting)
  - [47] Google Gemini generated image (lag features and cyclical time encoding)
  - [48] Seoul English website (Seoul map with bike stations image)
  - [49] Google Gemini generated image (hybrid model architectures)
- Always verify references exist before citing

## B.6 Domain Rules
- Avoid collisions with Ambika's thesis domain (German electricity market forecasting)
- Use transportation, bike-sharing, traffic, and urban mobility domain examples instead
- When explaining concepts (RNN, LSTM, etc.), use transport/mobility applications as examples

---

# PART C: THESIS STRUCTURE — Strict 6 Chapters

## C.1 Page Targets (~40 pages remaining for Ch 3–6, quality over quantity)

| Chapter | Title | Pages | Difficulty | Status |
|---------|-------|-------|------------|--------|
| Ch 1 | Introduction | ~10 | Easy | COMPLETE |
| Ch 2 | State of the Art | ~35-40 | Hardest | CONTENT COMPLETE |
| Ch 3 | Methodology and Dataset | ~8-10 | Medium (blueprint only) | DRAFTED (in .docx, needs fixes) |
| Ch 4 | Implementation | ~10-12 | Easy (from your own work) | IN PROGRESS (4.1–4.4.1 drafted) |
| Ch 5 | Evaluation | ~10-12 | Medium (results + discussion) | TO BE WRITTEN |
| Ch 6 | Conclusion and Future Work | ~5 | Easy | TO BE WRITTEN |

**Remaining budget: ~40 pages for Ch 3–6. Prioritise quality and conciseness over volume.**

## C.2 Chapter 1: Introduction (~10 pages) — STATUS: COMPLETE
Expanded from expose. Same structure but longer and more detailed.

### Sections:
- 1.1 Introduction
- 1.2 Motivation
- 1.3 Problem Statement
- 1.4 Proposed Methodology (CRISP-DM framework)
- 1.5 Research Objectives
- 1.6 Research Questions (4 RQs)
- 1.7 Contributions
- 1.8 Thesis Structure (6-chapter overview)

### Research Questions:
- RQ1: Can deep learning architectures (LSTM and TCN) effectively capture complex non-linear temporal patterns in Seoul's bike-sharing demand?
- RQ2: What is the quantifiable performance impact of time-aware feature engineering (cyclical encodings, lag features, rolling statistics)?
- RQ3: Do hybrid architectures combining different neural network paradigms outperform individual counterparts?
- RQ4: What is the most effective architectural combination, and what are trade-offs between accuracy, complexity, and efficiency?

## C.3 Chapter 2: State of the Art (~35-40 pages) — STATUS: CONTENT COMPLETE (pending humanization)

### Part 1: Background (~12-15 pages)
- 2.1 Time Series Forecasting — DONE
  - 2.1.1 Supervised and Deep Learning — DONE
  - 2.1.2 Recurrent Neural Networks (RNN) — DONE
  - 2.1.3 Long Short-Term Memory (LSTM) — DONE
  - 2.1.4 Gated Recurrent Unit (GRU) — DONE
  - 2.1.5 Temporal Convolutional Networks (TCN) — DONE
  - 2.1.6 Attention Mechanisms — DONE
  - 2.1.7 XGBoost — DONE
- 2.2 Feature Engineering for Time Series — DONE
- 2.3 Bike-Sharing Systems and Demand Forecasting — DONE

### Part 2: Related Work (~12-15 pages)
- 2.4 Related Work (organized by methodology category)
  - 2.4.1 Traditional and Statistical Approaches — DONE
  - 2.4.2 Machine Learning Approaches — DONE
  - 2.4.3 Deep Learning Approaches — DONE
  - 2.4.4 Hybrid and Attention-Based Approaches — DONE

### Part 3: Comparison and Summary (~5-8 pages)
- 2.5 Comparison of Approaches — ONE comparison table — DONE
- 2.6 Summary and Discussion — gap analysis, justification — DONE

### Key Rules for Chapter 2:
- Theoretical/foundational model formulas go HERE (LSTM gates, GRU gates, TCN convolution, attention, XGBoost objective) — see B.4 Formula Placement Rules
- Applied/implementation formulas (loss functions, feature engineering, CBAM, mixup) go in Chapter 4
- No per-topic advantages/disadvantages — compare ONCE at end
- Background = bring reader to same level before complex concepts
- Related work = organized by methodology category, NOT listed one by one
- Summary = explain why existing work is insufficient and why our approach is needed

## C.4 Chapter 3: Methodology and Dataset (~10 pages) — STATUS: DRAFTED IN .DOCX (needs fixes)

### CRITICAL RULE: Chapter 3 = BLUEPRINT / PLAN ONLY
- Professor's exact words: "methodology is a plan... without any action... no code, no library, no specific technology"
- Think of it like an architect's blueprint — rough plan with diagrams
- NO code, NO library names, NO specific technology names
- NO specific parameters or configurations
- Describe WHAT you will do, not HOW specifically with which tools
- Formulas go in Chapter 4 (implementation), NOT here
- Mostly diagrams, flowcharts, and high-level descriptions

### CONFIRMED STRUCTURE (aligned with professor's guidelines):
Chapter 3 has two natural halves:
  HALF 1 — Dataset (3.1): introduce the data used
  HALF 2 — Methodology/Blueprint (3.2–3.5): plan of what will be done

### Planned Sections:
- 3.1 Dataset Description
  - Seoul Bike Sharing Demand Dataset (UCI)
  - 8,760 hourly records, Dec 2017 – Nov 2018
  - 14 raw variables (1 target + 5 temporal + 8 weather)
  - Data quality: 100% complete, no missing values
  - Target distribution: right-skewed, mean 704.6 bikes/hour
  - Comparison to other bike-sharing datasets (D.C., etc.) — single city, system-wide aggregation
  - Figure: temporal distribution of rentals across the year [7]; target variable histogram [7]; Seoul bike stations map [48]

- 3.2 Data Preprocessing and Feature Engineering Plan (blueprint only)
  - Temporal train/test split rationale (80/20 chronological — why not random)
  - Scaling approach (standardization concept — no library names)
  - Sliding window concept for sequence creation (24h window → 1-step-ahead label)
  - Data leakage prevention strategy
  - Two-phase feature engineering plan:
    - Phase 1: expand from 14 raw → 64 candidate features (demand history, cyclical encodings, weather interactions)
    - Phase 2: optimise via correlation/multicollinearity analysis → 30 final features
  - Figure: preprocessing pipeline diagram; sliding window diagram; feature engineering pipeline [47]

- 3.3 Modelling Approach (blueprint only)
  - 1-step-ahead forecasting strategy — why this is fair vs same-time prediction
  - Individual model plan: LSTM and TCN as baselines (why these two)
  - Hybrid model plan: 5 hybrids combining complementary architectures
  - Why 7 specific architectures — justified by gap analysis in Ch2
  - Overall CRISP-DM methodology flowchart
  - Figure: CRISP-DM flowchart [6, 22]; 7-model selection diagram [49]

- 3.4 Evaluation Plan (blueprint only)
  - Metrics: R², RMSE, MAE — why these three
  - Generalisation assessment: train vs test gap analysis
  - Comparison framework: individual vs hybrid, complexity vs accuracy
  - Error analysis plan: by time of day, by weather condition
  - Figure: evaluation framework diagram

### Key Data Points for Chapter 3:
- Dataset: UCI Seoul Bike Sharing, 8,760 records, hourly, 14 raw features
- Split: 6,840 train (80%) / 1,752 test (20%), temporal/chronological
- Features: 14 raw → 64 engineered → 30 optimised (zero performance loss)
- 30 final features: 6 demand history, 3 temperature, 6 cyclical time, 3 rush hour, 6 weather, 6 categorical
- Demand history = 39.3% of Pearson |r| share (CORRECTED); demand_lag_1h r=0.908
- 7 models: 2 individual (LSTM, TCN) + 5 hybrid

## C.5 Chapter 4: Implementation (~10-15 pages) — STATUS: IN PROGRESS (4.1–4.4.1 drafted)

### CRITICAL RULE: Chapter 4 = SPECIFIC TECHNOLOGY & CODE
- This is where you name libraries, show code, list parameters
- Code screenshots of important scripts
- Specific hyperparameters, learning rates, epochs, etc.
- Formulas for loss functions, metrics, model equations go HERE
- Programming language, framework versions, hardware specs

### CRITICAL RULE: Chapters 4–6 = OUR OWN WORK
- From Chapter 4 onward, the content is based on our own implementation, experiments, and results
- Figures should primarily be author-generated: architecture diagrams, code screenshots, training curves, result plots, correlation heatmaps, prediction vs actual charts — all from our own codebase and experiments
- External figures (from papers/websites) are rarely needed here — this is our work, not literature
- Tables should show our own data: hyperparameters, model configurations, results, feature lists
- Every claim must be backed by our own experimental evidence (metrics, plots, code)
- Do NOT pad with generic theory — Chapter 2 already covered that. Chapters 4–6 are concise and evidence-driven
- Quality over quantity: a well-labelled figure or table replaces paragraphs of description

### Planned Sections:
- 4.1 Development Environment — DRAFTED
  - Python 3.12, TensorFlow/Keras 2.x, PyTorch 2.9, XGBoost 3.1, scikit-learn 1.7, pandas 2.3, NumPy 2.3
  - CPU-only (Intel Core, 16GB RAM), repository structure
  - Figures: repo directory tree screenshot (4.1.1), software stack table (Table 4.1.1)

- 4.2 Data Preprocessing Implementation — DRAFTED
  - StandardScaler (train-only fit), separate target scaler, sliding window (24,30)→1-step
  - 6,840 train / 1,752 test, 6,816 / 1,728 sequences, 20% validation from training
  - Data leakage safeguard (duplicate target column exclusion)

- 4.3 Feature Engineering Implementation — DRAFTED
  - Two-phase: 14→64→30 features, cyclical encoding formula, correlation/multicollinearity
  - 34 pairs at |r|>0.9 reduced to 2, demand history 39.3% Pearson |r| share (CORRECTED), demand_lag_1h r=0.908
  - Figures: correlation heatmap (4.3.1), feature importance chart (4.3.2), feature table (Table 4.3.1), Formula 4.3.1

- 4.4 Model Implementations (all 7 models)

  ### Writing Pattern for Each 4.4.X Subsection (learned from Ambika's thesis):
  Each model subsection follows this structure in flowing prose (NO numbered steps):
  1. Opening: reference Chapter 2 theory, state the compact formula for this model
  2. Architecture walkthrough: exact layers, units, activations, dropout — from our code
  3. Code screenshot figure: specific file + line numbers for the model definition
  4. Training config: optimiser, lr, epochs, callbacks — from our code
  5. Training callback/config code screenshot figure: specific file + line numbers
  6. Hyperparameter summary table
  7. Architecture diagram figure
  Target: ~400-500 words, 2-3 figures + 1 table + 1-2 formulas per model subsection

  - 4.4.1 LSTM Baseline — DRAFTED
    - 3 LSTM layers (128→64→32), BatchNorm+Dropout(0.3), Dense(64→32→1)
    - Keras Sequential API, Adam lr=0.001, batch 64, 150 epochs, patience 20
    - Formula: compact LSTM cell (4.4.1), MSE loss (4.4.2)
    - Figures: code screenshot lines 121-148 (4.4.1), callbacks lines 166-187 (4.4.2), architecture diagram (4.4.3)
    - Table: hyperparameter summary (Table 4.4.1)

  - 4.4.2 TCN Baseline — TO WRITE
    - 5 blocks [128,128,64,64,32], kernel 3, dilations [1,2,4,8,16], dropout 0.3, Dense(64→32→1)
    - PyTorch, Adam lr=0.001, weight decay 1e-5, batch 32, 150 epochs, gradient clip 1.0
    - Formula: dilated causal convolution (4.4.3), receptive field (4.4.4)
    - Figures: code screenshot of TemporalBlock + EnhancedTCN, architecture diagram
    - Code file: tcn/train_tcn_enhanced.py

  - 4.4.3 Hybrid TCN-LSTM — TO WRITE
    - TCN(5 blocks [128,128,64,64,32]) → LSTM(2×128) → Dense(128→64→1), 484,641 params
    - PyTorch, Adam lr=0.001, batch 32, 100 epochs, patience 15
    - No new formulas (combines TCN + LSTM from 4.4.1/4.4.2)
    - Code file: hybrid/train_hybrid.py

  - 4.4.4 TCN-GRU-Attention — TO WRITE
    - TCN(3 blocks [64,64,32]) → GRU(2×128) → MultiHeadAttention(4 heads, dim 128) → Dense(64→32→1), 294,177 params
    - PyTorch, Adam lr=0.001, batch 32, 100 epochs, gradient clip 1.0
    - Formula: scaled dot-product attention (4.4.5) — only if not already in Ch2
    - Code file: tcn_gru_attention/train_tcn_gru_attention.py

  - 4.4.5 TCN-CBAM-LSTM (Novel) — TO WRITE
    - TCN(5 blocks [64,64,64,32,32]) + CBAM(reduction 8, layers 3-5) → LSTM(2×128) → Dense(128→64→1), 330,382 params
    - AdamW lr=0.002, weight decay 1e-4, cosine annealing, dropout 0.2, batch 64, 150 epochs, patience 25
    - Formula: CBAM channel attention (4.4.6), CBAM spatial attention (4.4.7) — OUR novel contribution
    - Code file: tcn_cbam_lstm/train_tcn_cbam_lstm.py

  - 4.4.6 LSTM-XGBoost — TO WRITE
    - Stage 1: LSTM(2×128) → 64D features (50 epochs). Stage 2: 64D+30D=94D → XGBoost(500 trees, depth 6, lr 0.1)
    - 222,272 LSTM params + XGBoost trees
    - Formula: XGBoost objective with regularisation (4.4.8) — only if not in Ch2
    - Code file: lstm_xgboost/train_lstm_xgboost.py

  - 4.4.7 Multi-Scale TCN+LSTM — TO WRITE
    - Input proj(30→32) → 3 parallel TCN branches (kernels [2,3,5], [32,24] channels) → concat(72D) → LSTM(2×64) → Dense(32→1), 92,849 params
    - AdamW lr=0.001, weight decay 1e-3, cosine annealing, mixup α=0.1, noise 0.02, batch 64, 200 epochs, patience 30
    - Formula: multi-scale concatenation (4.4.9), mixup augmentation (4.4.10)
    - Code file: multi_scale_tcn/train_multi_scale_tcn_lstm.py

## C.6 Chapter 5: Evaluation (~10-15 pages) — STATUS: TO BE WRITTEN

### CRITICAL RULE: Discuss BOTH successes AND failures
Professor explicitly says: "Don't only talk about the good thing — talk about failures too"

### Planned Sections:
- 5.1 Evaluation Design
  - 1-step ahead prediction methodology
  - Why this is fair vs literature methods (same-time prediction is unfair)
  - Comparison with CUBIST paper (94% R² unfair vs our 86.67% fair)
  - Metrics: R², RMSE, MAE
  - Figure: 1-step ahead vs same-time prediction comparison diagram

- 5.2 Results
  - Complete results table (all 7 models ranked by Test R²)
  - Individual model results: LSTM (66.32%), TCN (75.69%)
  - Hybrid model results: TCN-LSTM (84.75%), TCN-GRU-Attention (85.58%), TCN-CBAM-LSTM (84.37%), LSTM-XGBoost (86.67%), Multi-Scale TCN+LSTM (88.83%)
  - Best model: Multi-Scale TCN + LSTM (88.83% R², only 92,849 parameters)
  - Figure: results comparison bar chart, R² ranking, RMSE comparison

- 5.3 Comparative Analysis
  - Individual vs Hybrid: hybrids consistently outperform individuals
  - Best hybrid (88.83%) vs best individual (75.69%) = +13.14% R² gap
  - Parameter efficiency: Multi-Scale TCN+LSTM wins (88.83% with only 92,849 params)
  - Complexity vs accuracy trade-offs across all 7 models
  - Why TCN-CBAM-LSTM is the novel contribution even without ranking first
  - **PROFESSOR REQUIREMENT — Ablation study:** Multi-Scale TCN alone (without LSTM) vs Multi-Scale TCN+LSTM — show how each component contributes to overall performance
  - Figure: scatter plot (parameters vs R²), model ranking visualization, ablation comparison chart

- 5.4 Error Analysis
  - Prediction vs actual plots for best models (Multi-Scale TCN+LSTM, LSTM-XGBoost)
  - Residual analysis
  - Error by time of day and weather condition
  - Figure: error distribution plots, prediction vs actual plots

- 5.5 Data Leakage Investigation
  - Context: early TCN experiments showed suspiciously high performance
  - Root cause: 'target' column included as input feature in early runs
  - Resolution: strict exclusion of both 'target' and 'Rented Bike Count' in all final models
  - All 7 final submitted models are confirmed clean
  - Lessons learned for reproducibility
  - Figure: code snippet showing the prevention strategy

- 5.6 Discussion of Results
  - Why Multi-Scale TCN+LSTM won: multi-kernel approach captures hourly, short-term, and half-day patterns; LSTM then models sequential dependencies across those multi-scale features
  - Why TCN-CBAM-LSTM is a novel contribution even though it did not top the leaderboard
  - Why attention (TCN-GRU-Attention) provides interpretability advantage
  - Limitations: individual LSTM and TCN underperform significantly (66-75%)
  - Failure analysis: why simpler models fell short
  - Feature importance findings: demand history = 39.3% Pearson |r| share (CORRECTED; old 78% was wrong)
  - Figure: attention weight visualization, feature importance chart

- 5.7 Comparison with Literature
  - Sathishkumar et al. (2020) GBM: R²=0.92 on same Seoul dataset, classical ML [7]
  - Sathishkumar et al. (2020) XGBoost: R²=0.91 on same dataset [7]
  - Aghe (2023) Stacking Regressor: R²=89.56% [40]
  - Our best (Multi-Scale TCN+LSTM): 88.83% using strict 1-step ahead temporal split
  - Key argument: literature results often use 5-fold CV or same-time prediction (inflated). Our methodology uses strict temporal split + 1-step ahead = deployable, fair evaluation
  - Professor's suggestion: note CUBIST 95% used same-time prediction — our evaluation is more rigorous

## C.7 Chapter 6: Conclusion and Future Work (~5 pages) — STATUS: TO BE WRITTEN

### CRITICAL RULE: Answer each RQ in at least HALF A PAGE
Professor explicitly says: "I would expect that you have like half of the page for each research question"

### Planned Sections:
- 6.1 Summary of Contributions
  - Comprehensive model development (7 architectures: 2 individual + 5 hybrid)
  - Best model: Multi-Scale TCN + LSTM (88.83% R², 92,849 params — most efficient)
  - Novel contribution: TCN-CBAM-LSTM (first use of CBAM attention for bike demand)
  - Hybrid architecture superiority demonstrated (+13.14% R² over best individual)
  - Feature engineering framework (14 raw → 64 engineered → 30 optimized, zero performance loss)
  - Rigorous evaluation methodology (1-step ahead temporal split, data leakage prevention)
  - Written in flowing prose, NOT bullet points

- 6.2 Answers to Research Questions (each ~half page, flowing prose)
  - RQ1 (P172): Can deep learning models capture nonlinear temporal patterns in Seoul bike demand? → Yes, but individual models insufficient; LSTM 66.32%, TCN 75.69%
  - RQ2 (P174): Is it useful to integrate attention mechanisms in hybrid architectures? → Partially; TCN-GRU-Attention 85.58% (3rd), TCN-CBAM-LSTM 84.37% (5th); attention adds value but does not dominate — non-attention Multi-Scale TCN+LSTM wins at 88.83%
  - RQ3 (P176): Do hybrid architectures outperform individual models? → Yes, conclusively; all 5 hybrids beat both individuals; weakest hybrid (84.37%) beats best individual (75.69%) by 8.68pp
  - RQ4 (P178): Best architecture and trade-offs? → Multi-Scale TCN+LSTM best overall (88.83%, 92,849 params); TCN-GRU-Attention best interpretability; TCN-CBAM-LSTM novel contribution; Hybrid TCN-LSTM worst ROI (484,641 params, only 4th place)

- 6.3 Limitations
  - Single city dataset (Seoul only)
  - System-wide aggregation (not station-level)
  - One year of data
  - CPU-based training (limited hyperparameter search)
  - Evening rush hour prediction remains challenging

- 6.4 Future Work
  - Probabilistic forecasting (confidence intervals)
  - Station-level prediction (graph neural networks)
  - Multi-step forecasting (3, 6, 12 hours ahead)
  - Transfer learning across cities
  - Real-time deployment pipeline
  - Explainability tools (SHAP values, attention visualization)

---

# PART D: MODELS SUMMARY

## D.1 The 7 Final Models — OFFICIAL RESULTS (submitted to professor Feb 2026)

These are the LOCKED final results as submitted to the professor. Do not change these numbers.

| Rank | Model | Type | Parameters | Test R² | Test RMSE | Test MAE |
|------|-------|------|------------|---------|-----------|----------|
| 1 | Multi-Scale TCN + LSTM | Hybrid (Multi-kernel TCN → LSTM) | 92,849 | 88.83% | 204.24 | 141.27 |
| 2 | LSTM-XGBoost | Two-Stage Ensemble | ~222K + XGB | 86.67% | 223.09 | 151.89 |
| 3 | TCN-GRU-Attention | Hybrid + Attention | ~294K | 85.58% | 231.98 | 151.91 |
| 4 | Hybrid TCN-LSTM | Hybrid (TCN → LSTM sequential) | ~487K | 84.75% | 238.60 | 158.25 |
| 5 | TCN-CBAM-LSTM | Hybrid + CBAM Attention | ~varies | 84.37% | 241.59 | 173.92 |
| 6 | TCN | Individual | ~180K | 75.69% | 301.28 | 226.63 |
| 7 | LSTM | Individual | ~110K | 66.32% | 354.58 | 260.48 |

## D.2 Model Descriptions (verified against actual code)

1. **LSTM** — 3 stacked LSTM layers (128→64→32), BatchNorm+Dropout(0.3) after each, Dense(64→32→1). TensorFlow/Keras. Tanh activation. Adam lr=0.001, batch 64, epochs 150, patience 20. File: lstm/train_lstm_enhanced.py

2. **TCN** — 5 temporal blocks, channels [128,128,64,64,32], kernel 3, dilations [1,2,4,8,16], dropout 0.3. Dense(64→32→1). PyTorch. Adam lr=0.001, weight decay 1e-5, batch 32, epochs 150, patience 20, gradient clip 1.0. File: tcn/train_tcn_enhanced.py

3. **Hybrid TCN-LSTM** — TCN(5 blocks [128,128,64,64,32]) → LSTM(2×128, dropout 0.3) → Dense(128→64→1). 484,641 params. PyTorch. Adam lr=0.001, batch 32, epochs 100, patience 15. File: hybrid/train_hybrid.py

4. **TCN-GRU-Attention** — TCN(3 blocks [64,64,32]) → GRU(2×128) → MultiHeadAttention(4 heads, dim 128, residual+LayerNorm) → Dense(64→32→1). 294,177 params. PyTorch. Adam lr=0.001, batch 32, epochs 100, patience 15, gradient clip 1.0. File: tcn_gru_attention/train_tcn_gru_attention.py

5. **TCN-CBAM-LSTM** — TCN(5 blocks [64,64,64,32,32]) + CBAM(reduction 8, applied to layers 3-5, channel+spatial attention) → LSTM(2×128) → LayerNorm → Dense(128→64→1). 330,382 params (2,093 CBAM). Novel contribution. AdamW lr=0.002, weight decay 1e-4, cosine annealing (T₀=20, T_mult=2), dropout 0.2, batch 64, epochs 150, patience 25, gradient clip 1.0. File: tcn_cbam_lstm/train_tcn_cbam_lstm.py

6. **LSTM-XGBoost** — Stage 1: LSTM(2×128, dropout 0.3) → Dense(64) feature extractor, trained 50 epochs. Stage 2: LSTM features(64D) + original features(30D) = 94D → XGBoost(500 trees, max_depth=6, lr=0.1, subsample 0.8, colsample 0.8, L1=0.1, L2=1.0). 222,272 LSTM params + XGBoost trees. File: lstm_xgboost/train_lstm_xgboost.py

7. **Multi-Scale TCN + LSTM** — BEST MODEL. Input projection(30→32) → 3 parallel TCN branches (kernels [2,3,5], each 2 levels [32,24], LightTemporalBlock with SpatialDropout) → concat(72D) → LayerNorm → LSTM(2×64) → LayerNorm → Dense(32→1). 92,849 params. AdamW lr=0.001, weight decay 1e-3, cosine annealing (T₀=20, T_mult=2), mixup α=0.1, noise std=0.02, dropout 0.3, batch 64, epochs 200, patience 30, gradient clip 1.0. File: multi_scale_tcn/train_multi_scale_tcn_lstm.py

## D.3 Key Findings
- Best model: Multi-Scale TCN + LSTM (88.83% R²) — most parameter-efficient at 92,849 params
- Hybrid models outperform individual models consistently
- Best hybrid beats best individual by +13.14% R² (88.83% vs 75.69%)
- TCN-CBAM-LSTM = novel contribution (CBAM for bike demand forecasting)
- TCN-GRU-Attention: best interpretability via attention weights
- Individual LSTM and TCN are baselines — included to justify need for hybrids
- All 7 models use same 30 features, 24-timestep sequences, same data split
- Feature engineering: demand history = 39.3% Pearson |r| share (CORRECTED); demand_lag_1h = strongest single predictor (r=0.908)

## D.4 Comparison with Literature Benchmarks
- Sathishkumar et al. (2020) GBM: R²=0.92 test — using same Seoul dataset, classical ML [7]
- Sathishkumar et al. (2020) XGBoost: R²=0.91 test — classical ML baseline [7]
- Aghe (2023) Stacking Regressor: R²=89.56% — classical ensemble [40]
- Our Multi-Scale TCN+LSTM: 88.83% — deep learning with fair 1-step ahead evaluation
- Note: Literature often uses 5-fold CV or same-time prediction (inflated). Our methodology uses strict temporal split + 1-step ahead = deployable, fair evaluation.

## D.5 Experimental / Non-Thesis Model Results (Session Mar 9, 2026)

These are results from experimental runs OUTSIDE the official thesis submission. Do not use these in thesis text. Recorded for reference only.

| Model | Config | Train R² | Test R² | Test RMSE | Test MAE |
|-------|--------|----------|---------|-----------|----------|
| TCN Enhanced (experimental) | seq=8, channels=[32,16,16], dropout=0.5, wd=2e-2, batch=64 | 93.05% | 81.92% | 260.47 | 198.89 |
| LSTM Enhanced (unchanged) | 30 features, seq=24, 3 LSTM layers | 95.66% | 75.76% | 300.84 | 218.80 |

Note: TCN official thesis result remains: Test R²=75.69%, RMSE=301.28, MAE=226.63 (19 features, train_tcn_basic.py).
The TCN Enhanced script (train_tcn_enhanced.py) was modified experimentally on Mar 9, 2026 (reduced capacity to approach ~82% R²).
A tuning script `tcn/tune_tcn_r2_target.py` was also created with 5 configs to explore further R² reduction without feature dropping.

---

# PART E: PROFESSOR'S FEEDBACK & ACTION ITEMS

## E.1 From Meeting 2 (Key Points)
- "Usually they use TCN before LSTM, not the other way around"
- "LSTM and GRU are basically the same thing" — combining makes less sense
- Praised proper 1-step ahead forecasting methodology
- Suggested re-implementing CUBIST paper with 5-fold to demonstrate its inflated results
- Search for ALL papers using Seoul Bike dataset to ensure no missed work
- Full draft needed by end of February 2026
- All coding must be complete by January 31, 2026

## E.2 From Professor — Pre-Writing Feedback (Feb 2026)
- "You can focus on the writing now."
- **MANDATORY for Chapter 5 (Evaluation):** "Report a breakdown evaluation: only Multi-Scale TCN without LSTM to show how each component contributed to the overall performance and report it in the Evaluation chapter."
- This means: run or report the Multi-Scale TCN component alone (no LSTM head) to show an ablation — how much does the LSTM add on top of the multi-scale TCN features?
- This is an **ablation study** requirement: Multi-Scale TCN alone vs Multi-Scale TCN+LSTM, demonstrating each component's contribution
- Must appear in Chapter 5 (Evaluation), likely in Section 5.3 (Comparative Analysis) or a dedicated ablation subsection

## E.3 From Writing Guidelines Lecture (Key Points)
- Chapter 2 is hardest, takes half of writing time, is half of thesis length
- Chapter 3 = BLUEPRINT ONLY (no code, no libraries, no specific technology)
- Chapter 4 = IMPLEMENTATION (code, libraries, parameters, formulas)
- Chapter 5 = must discuss failures, not just successes
- Chapter 6 = each RQ answered in at least half a page
- Plagiarism check: must be under 10-20% similarity
- New Declaration of Authorship form required (includes AI usage)
- Defense: 15 minutes presentation + 15 minutes Q&A, video demo recommended
- Grading: 25% defense + 75% written thesis

## E.4 Known Issues in Current .docx (User Must Fix)
- Garbage text "Object as Behaviour Function..." in GRU section
- Sentence fragment "Since then, due to its remarkable success..." in LSTM
- Merged subheadings in TCN section (Paras 237, 240)
- Duplicated "The output gate" in LSTM section
- Broken sentence "backward.recurrent nets" in RNN section
- Second-person "Your time series model" in section 2.1
- Missing citations in Paras 207, 226, 233
- Vary transition sentences (currently too mechanical/repetitive)

---

# PART F: CONTENT GENERATION WORKFLOW

## F.1 How We Write Each Section
1. Identify the section to write
2. Claude reads relevant source material (papers, code, data, existing content)
3. Claude drafts content in chat as flowing academic prose
4. Content follows ALL rules from Part B (no bullets, no bold, no AI patterns)
5. Include figure/formula/table placeholders inline where referenced in prose
6. For Ch4 model subsections: follow the writing pattern in C.5 (code screenshots with file+lines, architecture diagram, hyperparameter table)
7. User reviews, humanizes the text (adjusts phrasing, adds personal touch)
8. User inserts into Utkarsh_Master_Thesis.docx manually
9. Claude updates memory/tracking files if needed

## F.3 Quality Checklist Before Sharing
- [ ] No bullet points anywhere
- [ ] No bold/italic in body text
- [ ] No "Advantages/Disadvantages/Summary" structure
- [ ] No second person (you/your)
- [ ] All citations verified as real
- [ ] Figure placeholders included every 1-2 pages
- [ ] Natural, varied transitions between paragraphs
- [ ] Academic prose tone throughout
- [ ] Domain examples from transportation/bike-sharing (not electricity)

---

# PART G: CURRENT STATUS & NEXT STEPS

## G.1 Chapter Status Overview (Updated Mar 1, 2026)

| Chapter | Status | Notes |
|---------|--------|-------|
| Ch 1: Introduction | COMPLETE | In .docx, reviewed |
| Ch 2: State of the Art | IN .DOCX — FORMULA IMAGES ADDED | Content complete, ~38 minor text fixes remaining |
| Ch 3: Methodology & Dataset | IN .DOCX — PARTIALLY FIXED | ~10 remaining fixes (P362, P365, P386 most critical) |
| Ch 4: Implementation | IN .DOCX — ALL 7 MODELS WRITTEN | ~63 text fixes remaining, formula refs need Eqn→Figure update |
| Ch 5: Evaluation | IN .DOCX — COMPLETE | All sections written; 3 critical fixes: P543 broken sentence, P539 "Table 5.2"→"Table 5.2.1", P580 broken sentence fragment |
| Ch 6: Conclusion & Future Work | IN .DOCX — COMPLETE | Confirmed inserted by user Mar 9, 2026 |

## G.2 Next Steps (in order, updated Mar 9)
1. ~~Write Chapter 5~~ — DONE
2. ~~Write Chapter 6~~ — DONE
3. ~~Insert Chapter 6 into .docx~~ — DONE (confirmed Mar 9, 2026)
4. Fix Ch5 critical: P543 broken sentence, P539 "Table 5.2"→"Table 5.2.1", P580 broken sentence
5. Insert Ch5 figures: Figure 5.1.1, 5.2.1, 5.3.1, 5.5.1, Table 5.2.1, Table 5.7.1
6. Apply remaining text fixes: Ch2 (~38), Ch3 (~10), Ch4 (~63)
7. Update Ch4 formula references: "Equation 2.1.X" → "Figure 2.1.X.Y" (7 instances)
8. Final reference check (all citations)
9. Full draft by March 2–3 for plagiarism check, defense ~mid-March

## G.3 References Status
- [1]-[5]: LOCKED from original .docx
- [6]-[49]: Assigned in .docx (see Part B.4 for full list)
- [50]+: Available for new citations as we write Ch 3-6

---

# PART H: DATASET & FEATURE QUICK REFERENCE

## H.1 Dataset
- Source: UCI Seoul Bike Sharing Demand Dataset
- Period: Dec 1, 2017 – Nov 30, 2018
- Records: 8,760 (hourly)
- Target: Rented Bike Count (0-3,556)
- Split: 6,840 train (80%) / 1,752 test (20%), temporal/chronological

## H.2 Features (30 Final)
- 6 Demand History: lag_1h (r=0.908), lag_24h, lag_168h, rolling_3h_mean, rolling_24h_std, rolling_24h_max
- 3 Temperature: Temperature, temp_squared, temp_x_hour
- 6 Cyclical Time: hour_sin/cos, day_of_week_sin/cos, month_sin/cos
- 3 Rush Hour: is_rush_hour, is_evening_rush, time_of_day_evening
- 6 Weather: Humidity, Visibility, Solar Radiation, is_comfortable_weather, has_precipitation, bad_weather
- 6 Categorical: is_weekend, is_holiday, is_functioning, Season_Spring/Summer/Winter

## H.3 Feature Correlation Strength (Pearson |r| share — VERIFIED Feb 27, 2026)
**NOTE:** The previously claimed "78% demand history importance" was UNVERIFIED and incorrect.
Actual verified values computed from Pearson correlation with target:
- Demand History: **39.3%** (demand_lag_1h r=0.908 is strongest single feature)
- Temperature: 16.6%
- Weather: 13.6%
- Cyclical Time: 12.1%
- Categorical: 10.0%
- Rush Hour: 8.4%

XGBoost feature_importances_ (gain) from LSTM-XGBoost model give different breakdown:
- Temperature: 31.2%, Rush Hour: 30.9%, LSTM features: 18.5%, Cyclical Time: 7.0%
- Demand History only 3.5% in XGBoost (because LSTM already captured temporal patterns)

Figure 4.3.2 uses Pearson correlation shares (model-agnostic, computed during feature engineering).

---

# PART I: RECENT DECISIONS & CONVENTIONS (Feb 2026)

## I.1 Formula-as-Figure Convention (decided Feb 25)
- Ch2 formulas are inserted as images, captioned as Figures (not separate formula numbering)
- No formula numbers inside the images — only figure captions
- Example: Figure 2.1.3.2 contains the 6 LSTM gate equations
- Ch4 references to Ch2 formulas use "Figure 2.X.Y.Z" not "Equation 2.X.Y"

## I.2 Ch2 Formula Images Created (5 sets, 15 formulas total)
| Figure | Content | Formulas |
|--------|---------|----------|
| Figure 2.1.3.2 | LSTM Gate Equations | 6 (forget, input, candidate, cell update, output, hidden) |
| Figure 2.1.4.2 | GRU Gate Equations | 4 (reset, update, candidate, hidden) |
| Figure 2.1.5.2 | Dilated Causal Convolution | 1 |
| Figure 2.1.6.2 | Scaled Dot-Product & Multi-Head Attention | 2 |
| Figure 2.1.7.2 | XGBoost Ensemble & Regularised Objective | 2 |

3 formulas from original backlog were skipped (RNN hidden state, RNN output, TCN residual — not referenced in Ch4).

## I.3 Ch4 Formula Reference Updates Needed (7 instances)
These "Equation 2.X.Y" references in Ch4 must change to "Figure 2.X.Y.Z":
- P458: Equation 2.1.13 → Figure 2.1.5.2, Equations 2.1.3–2.1.8 → Figure 2.1.3.2
- P465: Equations 2.1.9–2.1.12 → Figure 2.1.4.2, Equations 2.1.15–2.1.16 → Figure 2.1.6.2
- P471: Equation 2.1.15 → Figure 2.1.6.2
- P502: Formulas 2.1.17–2.1.18 → Figure 2.1.7.2, Equations 2.1.3–2.1.8 → Figure 2.1.3.2

## I.4 Multi-Scale TCN+LSTM Architecture Diagram
- Created as draw.io XML: `temp_images/multi_scale_tcn_lstm_architecture.drawio`
- Verified code file: `multi_scale_tcn/train_multi_scale_tcn_lstm.py` (88.83% R², 92,849 params)
- Other variants (train_multi_scale_tcn.py, _regularized.py, _v2.py) are NOT the final model

## I.5 Figure 4.3.2 — Feature Importance (regenerated Feb 27)
- Previously used fabricated "78%" values
- Now uses verified Pearson correlation shares from actual data
- Generated by `papers/chapter4_figures.ipynb` (cell-8 and cell-9)
- Saved to `papers/figures/fig_4_3_2_feature_importance.png`
- Uses focus-attention design: navy for Demand History, muted gray for others

## I.6 Chapter Audit Change Lists (produced Feb 27)
Full change lists were produced for each chapter based on latest .docx content:
- Ch1: ~25 text fixes — confirmed complete by user
- Ch2: ~38 changes (3 figure numbering, 4 typos, 13 grammar, 7 tone, 6 citation format, 5 placeholders)
- Ch3: ~40 changes (20 grammar, 12 tone, 3 spelling, 2 figure numbering, 2 missing content)
- Ch4: ~63 changes (25 grammar, 18 tone, 5 spelling, 3 figure numbering, 7 formula refs, 2 version inconsistency)

## I.7 PyTorch Version Discrepancy
- P394 says PyTorch 1.9 but Table 4.1.1 placeholder says PyTorch 2.9 — needs verification and consistency
