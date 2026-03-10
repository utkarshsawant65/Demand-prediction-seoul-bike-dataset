# Chapter 2 — Formula Backlog

This document lists every theoretical/foundational formula that needs to be inserted into Chapter 2 (State of the Art) of the thesis. These formulas explain **what each model IS** mathematically. Applied/implementation formulas (loss functions, cyclical encoding, CBAM, mixup, receptive field) go in Chapter 4.

Numbering convention: Formula 2.section.sequence (e.g., Formula 2.1.1)

---

## Section 2.1.2 — Recurrent Neural Networks (RNN)

### Formula 2.1.1: RNN Hidden State Update
$$h_t = \tanh(W_h \cdot h_{t-1} + W_x \cdot x_t + b)$$

Where:
- $h_t$ = hidden state at time step $t$
- $x_t$ = input at time step $t$
- $W_h$, $W_x$ = weight matrices for hidden state and input
- $b$ = bias vector

**Context in prose:** This is the core recurrence relation that defines how an RNN processes sequential data. Introduce it when explaining why vanilla RNNs suffer from vanishing gradients (motivating LSTM/GRU).

### Formula 2.1.2: RNN Output
$$y_t = W_o \cdot h_t + b_o$$

Where:
- $y_t$ = output at time step $t$
- $W_o$ = output weight matrix
- $b_o$ = output bias

**Context in prose:** Mention briefly alongside Formula 2.1.1 — the output is a linear transformation of the hidden state.

---

## Section 2.1.3 — Long Short-Term Memory (LSTM)

Six formulas define the complete LSTM cell. Present them as a coherent set (all six together), not scattered.

### Formula 2.1.3: LSTM Forget Gate
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

### Formula 2.1.4: LSTM Input Gate
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

### Formula 2.1.5: LSTM Cell Candidate
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

### Formula 2.1.6: LSTM Cell State Update
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

### Formula 2.1.7: LSTM Output Gate
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

### Formula 2.1.8: LSTM Hidden State Output
$$h_t = o_t \odot \tanh(C_t)$$

Where (shared across all six):
- $\sigma$ = sigmoid activation function
- $\odot$ = element-wise (Hadamard) product
- $[h_{t-1}, x_t]$ = concatenation of previous hidden state and current input
- $W_f, W_i, W_C, W_o$ = weight matrices for each gate
- $b_f, b_i, b_C, b_o$ = bias vectors for each gate
- $C_t$ = cell state (long-term memory)
- $h_t$ = hidden state (short-term output)

**Context in prose:** The forget gate decides which information to discard from the cell state. The input gate and candidate determine what new information to store. The cell state update combines forgetting and adding. The output gate controls what portion of the cell state is exposed as the hidden state. Reference [13] Hochreiter & Schmidhuber 1997.

**Placement:** Section 2.1.3, after the prose explanation of LSTM architecture and before/alongside the LSTM architecture figure [24].

---

## Section 2.1.4 — Gated Recurrent Unit (GRU)

Four formulas define the GRU cell. Present as a coherent set.

### Formula 2.1.9: GRU Reset Gate
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

### Formula 2.1.10: GRU Update Gate
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

### Formula 2.1.11: GRU Candidate Hidden State
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)$$

### Formula 2.1.12: GRU Hidden State Update
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Where:
- $r_t$ = reset gate (controls how much past information to forget)
- $z_t$ = update gate (controls balance between old and new state)
- $\tilde{h}_t$ = candidate hidden state
- $\sigma$ = sigmoid activation, $\odot$ = element-wise product

**Context in prose:** The GRU merges the forget and input gates of the LSTM into a single update gate, and combines the cell state and hidden state into one. This reduces the parameter count while maintaining comparable performance. The reset gate allows the model to discard irrelevant past information. Reference [16] Cho et al. 2014, [27] Chung et al. 2014.

**Placement:** Section 2.1.4, after explaining how GRU simplifies LSTM, alongside the GRU cell figure [43].

---

## Section 2.1.5 — Temporal Convolutional Networks (TCN)

### Formula 2.1.13: Dilated Causal Convolution
$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t - d \cdot k}$$

Where:
- $y_t$ = output at time step $t$
- $w_k$ = filter weights (kernel parameters)
- $K$ = kernel size
- $d$ = dilation factor
- $x_{t - d \cdot k}$ = input at dilated offset (only past values, ensuring causality)

**Context in prose:** Unlike standard convolution, dilated convolution skips input values by a factor of $d$, enabling exponentially larger receptive fields without increasing the number of parameters. With $d = 1$, this reduces to standard convolution. Causality is ensured by only accessing time steps $\leq t$. Reference [14] Bai et al. 2018.

### Formula 2.1.14: Residual Connection
$$y = F(x) + x$$

Where:
- $F(x)$ = output of the temporal block (convolutions + nonlinearities)
- $x$ = input (skip connection)
- If dimensions of $F(x)$ and $x$ differ: $y = F(x) + W_s \cdot x$ (1x1 convolution for dimension matching)

**Context in prose:** Residual connections allow gradients to flow directly through the network, enabling training of deeper architectures. Each TCN block adds its learned transformation to its input, which stabilises training and mitigates vanishing gradients. Reference [29] He et al. 2016.

**Placement:** Section 2.1.5, after explaining the TCN architecture, alongside the stacked dilated convolution figure [44].

---

## Section 2.1.6 — Attention Mechanisms

### Formula 2.1.15: Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- $Q$ = query matrix (what we are looking for)
- $K$ = key matrix (what is available to attend to)
- $V$ = value matrix (the actual content)
- $d_k$ = dimensionality of the key vectors (scaling factor to prevent large dot products)
- $QK^T$ = dot-product similarity between queries and keys

**Context in prose:** The attention mechanism computes a weighted sum of values, where the weights are determined by the compatibility (dot product) between queries and keys. The scaling factor $\sqrt{d_k}$ prevents the dot products from growing too large, which would push the softmax into regions with vanishingly small gradients. Reference [15] Vaswani et al. 2017.

### Formula 2.1.16: Multi-Head Attention
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \cdot W^O$$

$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

Where:
- $h$ = number of attention heads
- $W_i^Q, W_i^K, W_i^V$ = learned projection matrices for head $i$
- $W^O$ = output projection matrix
- Each head attends to different representation subspaces independently

**Context in prose:** Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. Rather than performing a single attention function, the model projects queries, keys, and values $h$ times with different learned projections, applies attention in parallel, and concatenates the results. Reference [15] Vaswani et al. 2017.

**Placement:** Section 2.1.6, after the prose explanation of attention, before discussing how attention applies to time series. These two formulas should appear together.

---

## Section 2.1.7 — XGBoost

### Formula 2.1.17: XGBoost Ensemble Prediction
$$\hat{y}_i = \sum_{t=1}^{T} f_t(x_i)$$

Where:
- $\hat{y}_i$ = predicted value for sample $i$
- $T$ = total number of trees (boosting rounds)
- $f_t$ = the $t$-th regression tree
- Each tree is added sequentially to correct the residual errors of the ensemble so far

### Formula 2.1.18: XGBoost Regularised Objective Function
$$\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{t=1}^{T} \Omega(f_t)$$

$$\text{where } \Omega(f) = \gamma T_{\text{leaves}} + \frac{1}{2} \lambda \sum_{j=1}^{T_{\text{leaves}}} w_j^2$$

Where:
- $l(y_i, \hat{y}_i)$ = training loss (squared error for regression)
- $\Omega(f_t)$ = regularisation term for tree $t$
- $\gamma$ = penalty for number of leaves (controls tree complexity)
- $\lambda$ = L2 regularisation on leaf weights
- $T_{\text{leaves}}$ = number of terminal nodes (leaves)
- $w_j$ = weight (prediction score) of leaf $j$

**Context in prose:** XGBoost adds trees greedily, each one fitted to the negative gradient of the loss function (gradient boosting). The regularisation term $\Omega$ penalises both the number of leaves and the magnitude of leaf weights, preventing overfitting. This distinguishes XGBoost from standard gradient boosting by explicitly including structural regularisation in the objective. Reference [12] Chen & Guestrin 2016.

**Placement:** Section 2.1.7, after explaining the boosting concept and before the decision tree vs bagging vs boosting figure [46].

---

## Summary Table

| Section | Formula Numbers | Count | Topic |
|---------|----------------|-------|-------|
| 2.1.2 RNN | 2.1.1 – 2.1.2 | 2 | Hidden state, output |
| 2.1.3 LSTM | 2.1.3 – 2.1.8 | 6 | Forget, input, candidate, cell update, output gate, hidden state |
| 2.1.4 GRU | 2.1.9 – 2.1.12 | 4 | Reset, update, candidate, hidden state |
| 2.1.5 TCN | 2.1.13 – 2.1.14 | 2 | Dilated causal convolution, residual connection |
| 2.1.6 Attention | 2.1.15 – 2.1.16 | 2 | Scaled dot-product attention, multi-head attention |
| 2.1.7 XGBoost | 2.1.17 – 2.1.18 | 2 | Ensemble prediction, regularised objective |
| **Total** | | **18** | |

---

## Formulas NOT in Chapter 2 (they go in Chapter 4)

These are applied/configuration formulas. Listed here for clarity to avoid misplacement:

| Formula | Chapter 4 Section | Reason |
|---------|-------------------|--------|
| MSE loss function | 4.4.1 (LSTM) | Our chosen loss — implementation choice |
| Cyclical encoding (sin/cos) | 4.3 (Feature Engineering) | Our feature engineering method |
| Receptive field calculation | 4.4.2 (TCN) | Our specific TCN configuration |
| CBAM channel attention | 4.4.5 (TCN-CBAM-LSTM) | Our novel contribution |
| CBAM spatial attention | 4.4.5 (TCN-CBAM-LSTM) | Our novel contribution |
| Mixup augmentation | 4.4.7 (Multi-Scale TCN) | Our training technique |
| Cosine annealing LR | 4.4.5 / 4.4.7 | Our scheduler choice |
| Multi-scale concatenation | 4.4.7 (Multi-Scale TCN) | Our architectural design |
| StandardScaler (z-score) | 4.2 (Preprocessing) | Our preprocessing choice |
| Correlation coefficient | 4.3 (Feature Engineering) | Our feature selection method |

---

## Notes for Insertion

1. User will insert these as formula images in the .docx (using Word equation editor or LaTeX-rendered images)
2. Each formula must be referenced naturally in the prose BEFORE it appears
3. Use placeholder format in drafts: [Formula 2.1.X: description]
4. The LSTM set (6 formulas) and GRU set (4 formulas) should each appear as a unified block, not scattered across paragraphs
5. Chapter 2 content is already written in .docx — formulas need to be inserted into the existing prose at appropriate locations
