# Defense Plan — Utkarsh Sawant
# Last updated: March 5, 2026
# Status: PLANNED — changes may be needed, discuss before starting

---

## OVERVIEW

- Defense duration: 15 min presentation + 15 min Q&A
- Grading: 25% defense + 75% written thesis
- Professor recommends: video demo
- Defense date: ~mid-March 2026

---

## DELIVERABLE 1 — PowerPoint Presentation (15 slides, ~1 min each)

### Slide Plan

| Slide | Title | Key Content |
|---|---|---|
| 1 | Title | Thesis title, name, supervisor, institution, date |
| 2 | Motivation & Problem | Bike rebalancing challenge; 2,000+ systems globally; $1.6B market; Seoul map |
| 3 | Research Questions | All 4 RQs stated clearly — identical wording to thesis |
| 4 | Dataset | Seoul UCI, 8,760 records, demand distribution chart, right-skewed shape |
| 5 | Methodology | CRISP-DM, 80/20 temporal split, 1-step-ahead — WHY fairer than literature |
| 6 | Feature Engineering | 14→64→30 features; demand history 39.3%; lag_1h r=0.908; pipeline diagram |
| 7 | 7 Architectures Overview | Visual: 2 baselines + 5 hybrids; one-line description each; Figure 3.3.1 |
| 8 | Results Table | Full Table 5.2.1 — all 7 models ranked by Test R²; highlight Multi-Scale row |
| 9 | Results Chart | Bar chart Test R² all 7 models; two-tier gap visually clear; Figure 5.2.1 |
| 10 | Key Finding 1 — Hybrids Win | All 5 hybrids > both individuals; weakest hybrid +8.68pp; best +13.14pp |
| 11 | Key Finding 2 — Architecture > Size | Scatter plot params vs R²; Multi-Scale 92K/88.83% vs TCN-LSTM 484K/84.75% |
| 12 | Key Finding 3 — Ablation | Full 88.83%/92,849 vs TCN-only 88.43%/16,804; LSTM = 82% params, 0.4pp gain |
| 13 | Novel Contribution — CBAM | First CBAM use for bike-sharing; 2,093 params (0.6%); 84.37% result |
| 14 | Literature Comparison | CUBIST 95% (same-time) vs Aghe 89.56% (random split) vs ours 88.83% (strict) |
| 15 | Limitations & Future Work | Single city/year; system-wide only; future: station GNN, multi-step, probabilistic |

### Design Rules
- Max 5–6 points per slide or 1 chart/diagram — no text walls
- Reuse architecture diagrams already in thesis (Figures 4.4.X.1)
- Slides 8 and 9 are the centrepiece — results must be visually dominant
- Status: NOT STARTED

---

## DELIVERABLE 2 — Streamlit Dashboard (Video Demo)

### Purpose
Live interactive demo to play during the defense presentation.
Runs locally — no internet needed. No live model inference — loads saved predictions + metrics only.

### Tech Stack
- streamlit
- plotly (interactive charts — NOT matplotlib)
- pandas
- numpy

### Folder Structure (planned)
```
dashboard/
├── app.py
├── pages/
│   ├── 1_overview.py
│   ├── 2_features.py
│   ├── 3_results.py
│   ├── 4_predictions.py
│   └── 5_ablation.py
├── data/
│   └── (predictions CSVs + metrics JSONs)
└── assets/
    └── (architecture diagram PNGs)
```

### 5 Pages — Detailed Plan

#### Page 1 — Project Overview
- Thesis title, name, problem statement
- Metric cards: Best Model / Test R² 88.83% / 7 Architectures / 8,760 Records
- Demand distribution histogram
- Monthly demand bar chart
- Purpose: homepage / scene-setter

#### Page 2 — Feature Engineering
- Interactive correlation heatmap (30 features) — Figure 4.3.1
- Feature importance bar chart by category — Figure 4.3.2
- Table: 30 final features with Pearson |r| values
- 14→64→30 pipeline visual
- Purpose: shows data work; heatmap is interactive on hover

#### Page 3 — Model Results (Most Important)
- Full results table (Table 5.2.1) — color highlight top row
- Bar chart: Test R² all 7 models
- Scatter plot: Parameters vs Test R² (efficiency chart)
- Model selector dropdown → shows that model's hyperparameter table
- Purpose: centrepiece of demo; interactive model comparison

#### Page 4 — Predictions (The Live Demo Moment)
- Model selector dropdown: pick any of 7 models
- Date range slider: zoom into any week of test period
- Predicted vs actual demand line chart (updates with selection)
- Metric cards: R², RMSE, MAE for selected model
- Residual plot below
- Purpose: the "wow" moment — live interactive predictions

#### Page 5 — Ablation & Architecture Deep Dive
- Ablation: full model (88.83%, 92,849) vs TCN-only (88.43%, 16,804)
- Branch contribution pie chart: kernel-2 (29.3%), kernel-3 (35.7%), kernel-5 (35.0%)
- Architecture diagrams as images
- Error by hour of day bar chart (peak hour errors visible)
- Purpose: answers ablation question; professor mandated this

### Data Sources Already Available
| Dashboard need | Source |
|---|---|
| Test predictions | results/ folders — predictions CSVs per model |
| Model metrics | each model's *_metrics.json |
| Feature importance | papers/chapter4_figures.ipynb |
| Training history | each model's training_history*.csv |
| Architecture diagrams | thesis figures (PNG exports) |

### Build Priority Order
1. Page 3 — Results (show this first in defense)
2. Page 4 — Predictions (the live demo moment)
3. Page 1 — Overview (homepage)
4. Page 2 — Features (supporting)
5. Page 5 — Ablation (supporting)

### Status: NOT STARTED

---

## Q&A PREPARATION — Expected Questions

| Question | Key Answer |
|---|---|
| Why does Multi-Scale beat TCN-LSTM with 5x fewer params? | Multi-scale parallel branches capture 3 temporal resolutions simultaneously; design > size |
| CUBIST got 95%, you got 88.83% — worse? | No — CUBIST used same-time prediction + random split. Under our protocol CUBIST drops to ~66%. We proved this |
| Why combine TCN and LSTM — redundant? | TCN extracts local multi-scale features; LSTM models sequential dependencies across them. Ablation confirms 0.4pp gain |
| LSTM and GRU are the same — why use both? | We do NOT combine LSTM+GRU. TCN-GRU-Attention uses GRU+attention; TCN-CBAM-LSTM uses LSTM+spatial attention — different models |
| What does CBAM do in time-series? | Channel attention: which feature maps are most informative. Spatial attention: which timesteps matter most. 0.6% param overhead |
| What would you do differently? | Station-level (GNN), longer data, GPU for hyperparameter search, multi-step forecasting, probabilistic outputs |

---

## NOTES
- All plans above are DRAFT — changes may be needed
- Discuss with user before starting implementation of either deliverable
- PPT to be built in PowerPoint by user; Claude drafts slide content
- Dashboard to be built as code; Claude writes the Streamlit code
