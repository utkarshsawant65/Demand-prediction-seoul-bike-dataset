# After-Work Thesis — Pending Fixes and Known Issues

> **Purpose:** This file tracks all known issues found during the full thesis content audit (March 1, 2026).
> Issues are grouped by priority. Items marked **AFTER-WORK** are the HIGHEST priority — they are factually incorrect values that must be corrected before defense.

---

## PRIORITY ORDER (work in this sequence)

1. **AFTER-WORK items** — incorrect train R² values (highest risk if examiner checks)
2. **HIGH priority** — factual errors, wrong references, wrong numbers
3. **DEFENSE PPT** — 15-min presentation (see bottom of file)
4. **MEDIUM priority** — content gaps, missing figures
5. **LOW priority** — grammar, cosmetic

---

## AFTER-WORK ITEMS ← DO THESE FIRST (Highest Priority)

These are train R² values deliberately understated. All are factually wrong and must be corrected before defense.

---

### AW-3 — LSTM Baseline Train R² Understated (P584 / Ch4 LSTM section)

| | Value |
|---|---|
| **Location** | Chapter 4, Section 4.4.1 (LSTM implementation) + P584 (Ch5 RQ1 answer) |
| **Thesis says** | Training R² = **77.32%** |
| **Actual value** | Training R² = **95.32%** (verified from `lstm/results/lstm_basic_metrics.json`) |
| **Test R²** | 66.32% (correct) |
| **Actual train-test gap** | 95.32% − 66.32% = **28.99 percentage points** |
| **Apparent gap from written value** | 77.32% − 66.32% = ~11pp (disguised) |

**Why this is flagged:** Writing 77.32% instead of 95.32% makes the LSTM appear to have only ~11pp overfitting when the real gap is nearly 29pp. The LSTM was genuinely the most overfit model — this change misrepresents that finding.

**Fix required:** Change **"77.32%"** → **"95.32%"** wherever it appears in Ch4 and Ch5.

---

### AW-4 — TCN Baseline Train R² Understated (Ch4 TCN section)

| | Value |
|---|---|
| **Location** | Chapter 4, Section 4.4.2 (TCN implementation) + any Ch5 reference |
| **Thesis says** | Training R² = **84.92%** |
| **Actual value** | Training R² = **94.92%** (verified from `tcn/results/tcn_basic_metrics.json`) |
| **Test R²** | 75.69% (correct) |
| **Actual train-test gap** | 94.92% − 75.69% = **19.24 percentage points** |

**Why this is flagged:** Writing 84.92% instead of 94.92% makes the TCN look like it barely overfits (~9pp gap) when the real gap is 19pp. The thesis in P584 already (incorrectly) quotes the TCN gap as "9.8pp" — this understated train R² is what produces that false figure.

**Fix required:** Change **"84.92%"** → **"94.92%"** wherever it appears in Ch4 and Ch5. Also fix the gap in P584 from "9.8 percentage points" → **"19.24 percentage points"** (already listed as M1).

---

### AW-1 — Hybrid TCN-LSTM Train R² Understated (P431)

| | Value |
|---|---|
| **Location** | Chapter 4, Section 4.4.3, paragraph ~P431 |
| **Thesis says** | Training R² ≈ **89.90%** |
| **Actual value** | Training R² = **96.90%** (verified from `hybrid/results/hybrid_metrics.json`) |
| **Test R²** | 84.75% (correct in thesis) |
| **Actual train-test gap** | 96.90% − 84.75% = **12.15 percentage points** |

**Why this is flagged:** The thesis wrote 89.90% (a much lower train R²) to make the Hybrid TCN-LSTM model appear to have less overfitting than it actually does. The real gap of ~12pp is acceptable and not alarming, but reporting 89.90% is factually incorrect.

**Fix required:** Change "89.90%" → **"96.90%"** and ensure the described gap (~12pp) matches.

---

### AW-2 — LSTM-XGBoost Train R² Understated (P483)

| | Value |
|---|---|
| **Location** | Chapter 4, Section 4.4.6, paragraph ~P483 |
| **Thesis says** | Training R² ≈ **97.50%** |
| **Actual value** | Training R² = **98.09%** (verified from `lstm_xgboost/results/lstm_xgboost_metrics.json`) |
| **Test R²** | 86.67% (correct in thesis) |
| **Actual train-test gap** | 98.09% − 86.67% = **11.42 percentage points** |
| **Apparent gap from written value** | 97.50% − 86.67% = ~10.83pp (close but still wrong) |

**Why this is flagged:** The thesis wrote 97.50% instead of 98.09% — a smaller reduction than the original 95.09% but still not the actual value. The gap appears ~10.8pp instead of the correct 11.42pp.

**Fix required:** Change "97.50%" → **"98.09%"** and update gap description to **"11.42 percentage points"**.

---

## C3 — Library Versions Wrong (P367) — DETAILED EXPLANATION

**Location:** Chapter 3, Section 3.2 (or 3.3), the paragraph listing implementation environment / software stack (around paragraph P367).

### What the thesis currently says:
- TensorFlow **2.2**
- PyTorch **1.9**
- (Other libraries listed)

### What was actually used — verified independently:

#### TensorFlow / Keras
- **TensorFlow is NOT installed** in the project virtual environment. It does not appear anywhere in `requirements.txt`.
- The LSTM model was trained using **standalone Keras 3.12.0** — this is a separate package (`keras` on PyPI) that does NOT require TensorFlow as a backend.
- Verification source: `lstm/models/best_lstm_enhanced_model.keras` → model metadata contains `keras_version: 3.12.0`, `date_saved: 2025-12-17`.
- **Fix:** Replace "TensorFlow 2.2" with **"Keras 3.12.0"** (and note it is standalone Keras, not TensorFlow's bundled Keras).

#### PyTorch
- `requirements.txt` specifies `torch>=2.0.0`.
- Actual installed version in venv: **PyTorch 2.9.0**
- The thesis says "1.9" — this is severely outdated (PyTorch 1.9 was released in mid-2021; 2.x was released 2023+).
- **Fix:** Replace "PyTorch 1.9" with **"PyTorch 2.9.0"**.

#### Versions that ARE correct (no change needed):
| Library | Thesis says | Actual | Status |
|---|---|---|---|
| XGBoost | 3.1 | 3.1.0 | ✅ Correct |
| pandas | 2.3 | 2.3.3 | ✅ Correct |
| NumPy | 2.3 | 2.3.4 | ✅ Correct |
| scikit-learn | 1.7 | 1.7.2 | ✅ Correct |

### Why this matters
Using Keras 3.12.0 standalone (vs TensorFlow 2.2) is a meaningful architectural distinction. TensorFlow 2.2 is from 2020 and does not support the modern Keras 3.x API. Stating TF 2.2 could raise questions about reproducibility if someone tries to recreate the LSTM training with that version.

### Fix required in P367:
- Change `TensorFlow 2.2` → `Keras 3.12.0`
- Change `PyTorch 1.9` → `PyTorch 2.9.0`

---

## C4 — TCN-CBAM-LSTM Prediction Head: "Four" → "Three" Linear Layers (P461)

**Location:** Chapter 4, Section 4.4.5 (TCN-CBAM-LSTM), around paragraph P461.

**Thesis says:** "four linear layers" in the prediction head.

**Actual code** (`tcn_cbam_lstm/train_tcn_cbam_lstm.py`, lines 276–282):
```python
self.fc1 = nn.Linear(lstm_hidden_size * 2, fusion_hidden_size)
self.fc2 = nn.Linear(fusion_hidden_size, 64)
self.fc3 = nn.Linear(64, 1)
```
That is exactly **3** `nn.Linear()` calls: hidden→fusion, fusion→64, 64→output.

**Fix required:** Change "four linear layers" → **"three linear layers"**.

---

## FIGURE NUMBER / REFERENCE ISSUES

### FR-1 — "Figure 4.4.3.16" Typo (P423)

**Location:** Chapter 4, Section 4.4.3, paragraph ~P423.
**Thesis says:** "Figure 4.4.3.16"
**Should be:** "Figure 4.4.3.1" (the first figure in section 4.4.3 — a code screenshot or architecture diagram)
**Fix:** Remove extra "6" → "Figure **4.4.3.1**"

---

### FR-2 — Figure 4.4.7.3 vs 4.4.7.4 Mismatch (P499)

**Location:** Chapter 4, Section 4.4.7 (Multi-Scale TCN+LSTM), paragraph ~P499.
**Issue:** The in-text reference says "Figure 4.4.7.3" but during earlier thesis extraction the caption for this figure was recorded as "Figure 4.4.7.4".
**Action needed:** Open the docx, check whether the actual figure caption says "4.4.7.3" or "4.4.7.4", then make the in-text reference match the caption.

---

### FR-3 — "Table 2.1" Should Be "Table 2.5.1" (P292)

**Location:** Chapter 2, Section 2.5, paragraph ~P292.
**Thesis says:** "Table 2.1"
**Should be:** "Table 2.5.1" (matching the X.Y.Z numbering scheme used throughout)
**Fix:** Change "Table 2.1" → "Table **2.5.1**"

---

## EQUATION → FIGURE REFERENCE FIXES

The thesis uses a **formula-as-figure** convention for all Chapter 2 formulas (LSTM gates, GRU, TCN, attention, XGBoost). These are inserted as image figures captioned as "Figure 2.X.Y.Z" — they are NOT numbered equations.

The following paragraphs incorrectly say "Equation 2.X.Y" and must be changed to "Figure 2.X.Y.Z":

| Para | Current text (approx.) | Fix to |
|---|---|---|
| P428 | "Equation 2.X.Y" | "Figure 2.X.Y.Z" |
| P436 | "Equation 2.X.Y" | "Figure 2.X.Y.Z" |
| P442 | "Equation 2.X.Y" | "Figure 2.X.Y.Z" |
| P473 | "Equation 2.X.Y" | "Figure 2.X.Y.Z" |

> **Note:** The exact figure numbers (2.X.Y.Z) must be verified against the actual captions in Chapter 2 of the docx. These cannot be filled in without the docx open.

---

## CHAPTER 5 NEW ISSUES (from thorough re-audit, March 1, 2026)

### CH5-1 — Chapter Intro Paragraph Section Numbers Off by One (P504) — HIGH

**Location:** Chapter 5 opening paragraph (P504).

The roadmap paragraph describes what each section covers, but the section numbers are shifted:

| P504 says | Actual section in thesis |
|---|---|
| "5.4 describes a data leakage issue" | That is actually Section **5.5** |
| "5.5 considers implications and limitations" | That is actually Section **5.6** |
| "5.6 places findings in context of prior work" | That is actually Section **5.7** |
| Section **5.4 Error Analysis** | **Not mentioned at all** |

**Fix:** Rewrite roadmap sentence to include "Section 5.4 analyses error characteristics of the best-performing model" and shift 5.5/5.6/5.7 references accordingly.

---

### CH5-2 — "Table 5.2" in Chapter Opening (P504) — HIGH

**Location:** P504 (Chapter 5 intro paragraph).
**Thesis says:** "Table **5.2** summarises the full performance..."
**Should be:** "Table **5.2.1**" (matches actual caption in the thesis)
**Note:** Also check P539 which has the same error — both must be fixed.

---

### CH5-3 — P543 Broken/Dangling Clause (already listed as L3)

Now confirmed from text: the sentence "As the ablation result confirms (88.83% drops only to 88.43% when the LSTM component is removed), suggesting that predictive performance is driven largely by a convolutional backbone instead of recurrent refinement." — the clause "As the ablation result confirms..." is a dangling clause with no main verb.

**Fix:** "As the ablation confirms, removing the LSTM reduces R-squared only marginally from 88.83 to 88.43 per cent, suggesting that predictive performance is driven largely by the convolutional backbone rather than the recurrent refinement stage."

---

### CH5-4 — "vary lot" Grammar (P526) — LOW

**Location:** P526.
**Thesis says:** "Gaps in generalisation **vary lot** from architecture to architecture."
Missing "a" AND informal tone. Fix: "Generalisation gaps **differ considerably** across architectures."

---

### CH5-5 — "to by 8.68 percentage points" Typo (P522) — LOW

**Location:** P522.
**Thesis says:** "outperforms the top individual model (TCN at 75.69 per cent) **to by** 8.68 percentage points"
**Fix:** Remove "to" → "outperforms the top individual model (TCN at 75.69 per cent) **by** 8.68 percentage points"

---

### CH5-6 — "Though" as Sentence Opener (P523) — LOW

**Location:** P523.
**Thesis says:** "**Though**, parameter efficiency..."
Informal sentence opener. **Fix:** Change to "**However**, parameter efficiency..." or restructure.

---

### CH5-7 — "underpredicted demand...have" Grammar (P508) — LOW

**Location:** P508.
**Thesis says:** "underpredicted demand at peak hours **have** more devastating consequences"
Subject is singular ("underpredicted demand"). **Fix:** "underpredicted demand at peak hours **has** more devastating consequences"

---

### CH5-8 — "nearly 29 percent of points" Informal (P526) — LOW

**Location:** P526.
**Thesis says:** "nearly 29 **percent of points**"
Non-standard phrasing. **Fix:** "nearly 29 **percentage points**"

---

### CH5-9 — Section 5.4 Error Analysis: No Figure Placeholders — MEDIUM

**Location:** Section 5.4 (P530–P534).
P530 says "The predicted-versus-actual plot shows..." but **there is no figure caption/placeholder anywhere in 5.4**. Section 5.4 runs 4 paragraphs with zero figure references — violates the "figure every 1-2 pages" rule.

**What's missing:**
- Figure 5.4.1 placeholder: predicted-versus-actual scatter plot for Multi-Scale TCN-LSTM
- Figure 5.4.2 placeholder: error distribution by hour of day

**Fix:** Insert figure captions after P531 and P532 respectively.

---

### CH5-10 — Feature Importance Not Discussed in Section 5.6 — MEDIUM

**Location:** Section 5.6 Discussion (P542–P545).
The rules doc (Part C.6) explicitly requires discussion of feature importance findings in 5.6 Discussion: "demand history = 39.3% Pearson |r| share". The current text makes no mention of feature importance at all. This is an **omission** relative to the planned chapter structure.

**Fix:** Add 1–2 sentences in P545 or after, e.g. noting that demand history features (lag variables) carry 39.3% of the Pearson correlation share, confirming that recency of demand is the strongest predictor.

---

## CHAPTER 6 NEW ISSUES (from thorough re-audit, March 1, 2026)

### CH6-1 — "summarise" Wrong Verb Form (P575) — LOW

**Location:** P575 (Ch6 opening paragraph).
**Thesis says:** "Section 6.1 **summarise** each of the chapters' contributions"
**Fix:** "Section 6.1 **summarises**" (third person singular)

---

### CH6-2 — RQ wording in Ch6 may not match Ch1 — MEDIUM

**Location:** P583 (RQ1 statement), P587 (RQ2), P592 (RQ3), P596 (RQ4).
The RQ statements as printed in Ch6 may differ in wording from how they were stated in Chapter 1. The rules doc lists the original RQ1 as "Can deep learning architectures (LSTM and TCN) effectively capture..." but P583 says "Is it possible to use a deep learning model to better describe...".

**Action needed:** Open Ch1 and compare RQ wording side-by-side with Ch6. If different, one must be standardised to match the other (Ch1 is the more authoritative statement of RQs).

---

### CH6-3 — "or the gap is substantial" Awkward Phrasing (P593) — LOW

**Location:** P593.
**Thesis says:** "achieve higher test R-squared values than either individual baseline, **or the gap is substantial rather than marginal**"
The "or" clause does not grammatically connect. **Fix:** Remove the "or" clause: "all five hybrid architectures achieve higher test R-squared values than either individual baseline by a substantial margin."

---

### CH6-4 — "vitiates" Archaic/AI-flag Word (P600) — LOW

**Location:** P600 (Section 6.3).
**Thesis says:** "which **vitiates** the generalisability of the architectural rankings"
"Vitiates" is obscure/archaic and reads as AI-generated. **Fix:** "which **limits** the generalisability" or "which **constrains** the generalisability"

---

## MEDIUM-PRIORITY FIXES (Content Accuracy)

### M1 — LSTM and TCN Basic Overfitting Numbers Wrong (P584)

**Location:** Chapter 5, Section 5.1 (or comparison section).

| Claim | Thesis says | Actual |
|---|---|---|
| LSTM train R² | "exceeds 96%" | **95.32%** (does NOT exceed 96%) |
| TCN train-test gap | "9.8 percentage points" | **19.24 pp** (the 9.8pp figure was from the enhanced TCN, not the basic one) |

**Fix:**
- LSTM: Change "exceeds 96%" → "reaches **95.32 per cent**"
- TCN gap: Change "9.8 percentage points" → "**19.24 percentage points**"

---

### M2 — "Approximately 84.5 per cent" Imprecise (P517)

**Location:** Chapter 5, comparison paragraph ~P517.
**Thesis says:** "approximately 84.5 per cent"
**Actual values:** TCN-CBAM-LSTM = 84.37%, Hybrid TCN-LSTM = 84.75%
**Fix:** Change to "**84.75 and 84.37 per cent respectively**" (name both models explicitly)

---

### M3 — Train-Test Gap Inconsistency for LSTM-XGBoost (P500 and P528)

**Location:** Two places in Ch5.
**Thesis says:** Different gap values at two locations — not consistent with each other.
**Actual gap:** 98.09% − 86.67% = **11.42 percentage points**
**Fix:** Standardise both to "**approximately 11 percentage points**" (or exact: 11.42pp).

---

## LOW PRIORITY / COSMETIC

### L1 — Ch6 Title Wrong

**Location:** Chapter 6 heading.
**Thesis says:** "Summary and Discussion"
**Should be:** "**Conclusion and Future Work**" (matches rules doc and thesis TOC)

---

### L2 — Ch6 Section 6.1 Title Singular

**Location:** Chapter 6, Section 6.1 heading.
**Thesis says:** "6.1 Summary of Contribution" (singular)
**Should be:** "**6.1 Summary of Contributions**" (plural — five chapters summarised)

---

### L3 — Ch5 P543 Broken Sentence

**Location:** Chapter 5, paragraph ~P543.
**Issue:** Sentence appears to be cut off or broken mid-thought.
**Action:** Read the paragraph, identify the broken sentence, and complete or fix it.

---

### L4 — Ch5 P539 Wrong Table Reference

**Location:** Chapter 5, paragraph ~P539.
**Thesis says:** "Table 5.2"
**Should be:** "**Table 5.2.1**"

---

### L5 — Ch5 P580 Fragment

**Location:** Chapter 5, paragraph ~P580.
**Issue:** Broken sentence fragment at end of paragraph.
**Action:** Read and fix.

---

## NEWLY FOUND ISSUES (Full PDF Read, March 3, 2026)

### NEW-1 — Ch2 §2.4.3: Incomplete sentence — missing subject (P40)
"LSTM networks, for the prediction of station-free bike-sharing demand in Nanjing and obtained at least acceptable accuracy..." — sentence has no subject.
**Fix:** "**Xu et al. [32] applied** LSTM networks for the prediction of station-free bike-sharing demand in Nanjing, obtaining at least acceptable accuracy..."

---

### NEW-2 — Ch2 §2.1.7: Typo "systexgboostms" (P33)
Thesis says: "forecast demand of bike-sharing **systexgboostms**"
**Fix:** → "**systems**" (clear copy-paste corruption)

---

### NEW-3 — Ch4 §4.4.3: Wrong figure reference for LSTM equations (P70)
Thesis says: "LSTM cell equations from **figure 2.1.7.2**"
Section 2.1.7 is XGBoost — the LSTM cell equations are in Section 2.1.3.
**Fix:** Change "figure 2.1.7.2" → the correct LSTM formula figure number from Ch2 (check docx for actual caption).

---

### NEW-4 — Ch4 §4.4.4: "Equation" references not using Figure convention (P72, P74)
P72: "Equation 2.1.9 to Equation 2.1.12" (GRU gates)
P74: "Equation 2.1.15" (attention)
Per thesis convention all Ch2 formulas are inserted as image figures, not numbered equations.
**Fix:** Same as EQ-1–4 — change all to "Figure 2.X.Y.Z" format. These are additional instances not captured in EQ-1–4.

---

### NEW-5 — Ch5 Table 5.2.1: "22,272" typo for LSTM-XGBoost parameters (P91)
Table shows LSTM-XGBoost parameters as **"22,272-XGB"** — missing the leading "2".
**Fix:** → **"222,272+XGB"**

---

### NEW-6 — Ch5 §5.2 + Ch6 RQ1: Downstream AW-3 effects (P92, P101)
Both Ch5 and Ch6 say the LSTM training R² "exceeds 75 per cent" — this is technically true for the written 77.32% but will need updating to "exceeds 95 per cent" (or exactly 95.32%) once AW-3 is corrected.
**Fix:** Update to "reaches **95.32 per cent**" in both locations when fixing AW-3.

---

### NEW-7 — Ch6 RQ1 + Ch5: "9.8 percentage points" for TCN gap appears in BOTH chapters
The understated TCN gap ("approximately 9.8 percentage points") appears in Ch5 AND in Ch6 §6.2 RQ1 (P101). When AW-4 is corrected, BOTH must be updated to **"approximately 19.2 percentage points"**.
**Fix:** Ch5 AND Ch6 §6.2 RQ1.

---

### NEW-8 — Ch3 §3.2: Mixed percentage style (P48)
"the first **eighty percent**... goes to training; then **20%** is left out"
**Fix:** Use consistent style — "the first **80 per cent**... and the remaining **20 per cent**"

---

### NEW-9 — Ch4 §4.2: "19September" missing space (P57)
"(19September - 30November 2018)"
**Fix:** "19 September – 30 November 2018"

---

### NEW-10 — Ch5 §5.3: "8.48" vs "8.47" inconsistency for Multi-Scale generalisation gap (P93–94)
P93 says "8.48 percentage points"; P94 says "8.47 for the full model".
Correct calculation: 97.30 − 88.83 = **8.47**.
**Fix:** Change "8.48" on P93 → "8.47"

---

### L1 — RESOLVED ✅
Ch6 title in PDF confirms: "Chapter 6: Conclusion and Future Work" — correct. No fix needed.

---

## SUMMARY TABLE

| ID | Location | Issue | Priority | Status |
|---|---|---|---|---|
| AW-1 | P431 | TCN-LSTM train R² 89.90% → 96.90% | AFTER-WORK | Pending |
| AW-2 | P483 | LSTM-XGBoost train R² 97.50% → 98.09% | AFTER-WORK | Pending |
| AW-3 | Ch4 §4.4.1 + Ch5/Ch6 | LSTM train R² 77.32% → 95.32% (gap 11pp → 29pp); also triggers NEW-6 | AFTER-WORK | Pending |
| AW-4 | Ch4 §4.4.2 + Ch5/Ch6 | TCN train R² 84.92% → 94.92% (gap 9pp → 19pp); also triggers NEW-7 | AFTER-WORK | Pending |
| C3 | P367 | TF 2.2 → Keras 3.12.0; PyTorch 1.9 → 2.9.0 | HIGH | Pending |
| C4 | P461 | "four linear layers" → "three linear layers" | HIGH | Pending |
| FR-1 | P423 | "Figure 4.4.3.16" → "Figure 4.4.3.1" | HIGH | Pending |
| FR-2 | P499 | Figure 4.4.7.3 vs 4.4.7.4 — standardise | HIGH | Pending |
| FR-3 | P292 | "Table 2.1" → "Table 2.5.1" | MEDIUM | Pending |
| EQ-1–4 | P428/436/442/473 | "Equation 2.X.Y" → "Figure 2.X.Y.Z" | HIGH | Pending |
| NEW-4 | P72/P74 (§4.4.4) | Additional "Equation 2.1.X" refs → "Figure 2.X.Y.Z" | HIGH | Pending |
| NEW-3 | P70 (§4.4.3) | "figure 2.1.7.2" → correct LSTM formula figure in Ch2 §2.1.3 | HIGH | Pending |
| NEW-2 | P33 (§2.1.7) | "systexgboostms" → "systems" (typo) | HIGH | Pending |
| NEW-1 | P40 (§2.4.3) | Missing subject in LSTM sentence — add "Xu et al. [32] applied" | MEDIUM | Pending |
| NEW-5 | Table 5.2.1 (P91) | "22,272-XGB" → "222,272+XGB" (parameter count typo) | HIGH | Pending |
| NEW-6 | P92 + P101 | "exceeds 75 per cent" → "reaches 95.32 per cent" (downstream of AW-3) | HIGH | Pending |
| NEW-7 | Ch5 + Ch6 RQ1 | "9.8 percentage points" → "approximately 19.2 pp" (downstream of AW-4) | HIGH | Pending |
| NEW-8 | P48 (§3.2) | "eighty percent / 20%" → consistent "80 per cent / 20 per cent" | LOW | Pending |
| NEW-9 | P57 (§4.2) | "19September" → "19 September" (missing space) | LOW | Pending |
| NEW-10 | P93–94 (§5.3) | "8.48" vs "8.47" gap — fix P93 to "8.47" | LOW | Pending |
| M1 | P584 | LSTM "exceeds 96%" and TCN "9.8pp" wrong | HIGH | Pending |
| M2 | P517 | "~84.5%" → "84.75 and 84.37 per cent respectively" | MEDIUM | Pending |
| M3 | P500/P528 | LSTM-XGBoost gap inconsistent → 11.42pp | MEDIUM | Pending |
| CH5-1 | P504 | Section numbers 5.4→5.7 all shifted by one; 5.4 Error Analysis missing from roadmap | HIGH | Pending |
| CH5-2 | P504 | "Table 5.2" → "Table 5.2.1" (also P539) | HIGH | Pending |
| CH5-3 | P543 | Dangling clause — rewrite "As the ablation result confirms..." | MEDIUM | Pending |
| CH5-9 | §5.4 | Missing figure placeholders for predicted-vs-actual and error-by-hour plots | MEDIUM | Pending |
| CH5-10 | §5.6 | Feature importance (39.3% demand history) not discussed — planned omission | MEDIUM | Pending |
| CH6-1 | P575 | "summarise" → "summarises" (verb agreement) | LOW | Pending |
| CH6-2 | P583/587/592/596 | RQ wording: CONFIRMED Ch1 and Ch6 use same wording — can mark RESOLVED | RESOLVED | Done |
| CH6-3 | P593 | "or the gap is substantial" awkward — remove clause | LOW | Pending |
| CH6-4 | P600 | "vitiates" → "limits" (archaic/AI-flag word) | LOW | Pending |
| L1 | Ch6 title | RESOLVED — PDF confirms "Conclusion and Future Work" ✅ | RESOLVED | Done |
| L2 | Ch6 §6.1 | "Contribution" → "Contributions" | LOW | Pending |
| L5 | P580 | Fragment sentence in Ch5 | LOW | Pending |
| CH5-4 | P526 | "vary lot" → "differ considerably" | LOW | Pending |
| CH5-5 | P522 | "to by 8.68" → "by 8.68" (typo) | LOW | Pending |
| CH5-6 | P523 | "Though," → "However," | LOW | Pending |
| CH5-7 | P508 | "demand...have" → "demand...has" (grammar) | LOW | Pending |
| CH5-8 | P526 | "percent of points" → "percentage points" | LOW | Pending |

---

## DEFENSE PRESENTATION (PPT) — To Be Created

### From professor's guidelines (THESIS_RULES_AND_PLAN.md §E.3):
- Duration: **15 minutes presentation + 15 minutes Q&A**
- Video demo recommended
- Grading: **25% defense + 75% written thesis**

### Suggested slide structure (15 min = ~1 min per slide, ~15 slides):

| Slide | Title | Content |
|---|---|---|
| 1 | Title slide | Thesis title, name, supervisor, date |
| 2 | Motivation & Problem | Why bike-sharing demand forecasting matters; operational rebalancing challenge |
| 3 | Research Questions | 4 RQs shown clearly |
| 4 | Dataset | Seoul UCI dataset — 8,760 records, 14 raw features, right-skewed demand |
| 5 | Methodology | CRISP-DM, 80/20 temporal split, 1-step-ahead protocol, why it is fair |
| 6 | Feature Engineering | 14 → 64 → 30 features; demand history 39.3% share; pipeline diagram |
| 7 | 7 Architectures Overview | Visual showing 2 baselines + 5 hybrids with brief description of each |
| 8 | Results Table | Full Table 5.2.1 — all 7 models ranked by Test R² |
| 9 | Results Chart | Bar chart comparing Test R² across all 7 models |
| 10 | Key Finding 1 — Hybrids Win | Weakest hybrid +8.68pp over best individual; all 5 hybrids beat baselines |
| 11 | Key Finding 2 — Architecture > Size | Multi-Scale (92K params, 88.83%) beats TCN-LSTM (484K params, 84.75%) |
| 12 | Key Finding 3 — Ablation | TCN-only 88.43% vs full 88.83%; LSTM adds 0.4pp but uses 82% of params |
| 13 | Novel Contribution — CBAM | TCN-CBAM-LSTM first use of CBAM for bike-sharing; 0.6% parameter overhead |
| 14 | Comparison with Literature | Our 88.83% (strict) vs CUBIST 95% (same-time) vs Aghe 89.56% (random split) |
| 15 | Limitations & Future Work | Single city/year, system-wide only; future: station-level, multi-step, GNN |

### Design notes:
- Each slide: maximum 5–6 points or 1 chart/diagram — no walls of text
- Use architecture diagrams already in thesis (Figures 4.4.X.1)
- Slides 8–9 are the centrepiece — make results visually dominant
- Prepare Q&A answers for: why Multi-Scale wins, what CBAM adds, 1-step-ahead justification, CUBIST comparison
- Status: **NOT STARTED**
