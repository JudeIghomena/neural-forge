# Time-Series & Sequential Models — Deep Reference

## Architecture Taxonomy

```
Sequential Models
├── RNNs (stateful, sequential)
│   ├── Vanilla RNN (gradient vanishing)
│   ├── LSTM (gated memory)
│   └── GRU (simplified LSTM)
├── CNNs for sequences
│   ├── TCN (Temporal Convolutional Network)
│   └── WaveNet (dilated causal convolutions)
├── Transformers for sequences
│   ├── Vanilla Transformer + positional encoding
│   ├── Informer (sparse attention, long sequences)
│   ├── PatchTST (patching for time-series)
│   └── Temporal Fusion Transformer (multi-horizon)
└── State Space Models
    ├── S4 (structured state space)
    ├── Mamba (selective SSM)
    └── Mamba-2 (SSD)
```

---

## LSTM

### Equations
$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$\tilde{c}_t = \tanh(W_c[h_{t-1}, x_t] + b_c) \quad \text{(candidate cell)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell update)}$$
$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(c_t)$$

```python
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [B, T, F]
        out, (h_n, c_n) = self.lstm(x)
        return self.head(out[:, -1, :])  # Use last timestep
```

---

## TCN (Temporal Convolutional Network)

Dilated causal convolutions achieve long receptive fields with $O(\log n)$ layers.

$$\text{Receptive field} = 1 + 2(k-1)\sum_{l=0}^{L-1} d^l = 1 + 2(k-1)\frac{d^L-1}{d-1}$$

```python
class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # Causal padding
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               padding=pad, dilation=dilation)
        self.net = nn.Sequential(
            self.conv1, nn.Chomp1d(pad), nn.ReLU(), nn.Dropout(0.2),
            self.conv2, nn.Chomp1d(pad), nn.ReLU(), nn.Dropout(0.2)
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
```

---

## Mamba (Selective State Space Model)

**Key innovation**: input-dependent (selective) SSM parameters — $B, C, \Delta$ are functions of $x$.

### SSM Core
$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
$$y_t = C h_t$$

where $\bar{A} = \exp(\Delta A)$, $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\Delta B$ (ZOH discretization).

**Selectivity**: $\Delta, B, C = \text{Linear}(x)$ — allows the model to selectively remember or forget.

```
Why Mamba beats Transformer for long sequences:
- O(L) compute vs O(L²) for full attention
- O(1) inference memory (stateful RNN-like)
- Hardware-aware parallel scan (CUDA kernel)
- Matches Transformer quality at 1M+ context lengths
```

---

## Temporal Fusion Transformer (Multi-horizon Forecasting)

Best for: multi-horizon forecasts with mixed static/temporal features + interpretability.

Architecture components:
1. **Variable Selection Networks** — learn which inputs matter per timestep
2. **LSTM encoder-decoder** — captures local temporal patterns
3. **Multi-head attention** — captures long-range dependencies
4. **Quantile outputs** — predict 10th, 50th, 90th percentiles

Loss: Quantile loss (pinball loss)
$$\mathcal{L}_q(y, \hat{y}) = q \max(y - \hat{y}, 0) + (1-q)\max(\hat{y} - y, 0)$$

---

## PatchTST

Channel-independent + patching approach. State-of-the-art for many time-series benchmarks.

```
1. Segment time series into patches of size P
2. Each channel processed independently (univariate)
3. Patches → linear embedding → Transformer
4. Achieves linear attention complexity in effective sequence length
```

$$L_{\text{patches}} = \left\lfloor\frac{L - P}{S}\right\rfloor + 1, \quad S = \text{stride}$$

---

## Anomaly Detection Approaches

| Method | Type | When to use |
|---|---|---|
| Reconstruction-based (AE/VAE) | Unsupervised | No labels |
| Forecasting-based (LSTM) | Unsupervised | Sequential data |
| OCSVM + deep features | Semi-supervised | Few anomaly samples |
| Transformer + window attention | Unsupervised | Multivariate, long sequences |

### Anomaly Score
$$s_t = \|x_t - \hat{x}_t\|^2 \quad \text{(reconstruction error)}$$

Threshold via: rolling statistics, Extreme Value Theory (EVT), or learned threshold.

---

## Key Practical Tips

1. **Normalize per feature** (Z-score or MinMax) — time series are non-stationary
2. **Handle missing values**: forward fill + mask token, not mean imputation
3. **Train/val/test split**: always chronological, never random
4. **Seasonality**: include Fourier features or use seasonal decomposition (STL)
5. **Horizon matters**: TFT for multi-horizon, LSTM/TCN for single-step
6. **Lookback window**: generally 2-5× the forecast horizon
7. **Stationary test**: ADF test before choosing architecture
