# NLP & LLMs — Deep Reference

## Transformer Architecture (Full)

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # Flash Attention (PyTorch 2.0+)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(attn)
```

---

## Positional Encodings

### Sinusoidal (original)
$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

### RoPE (Rotary Position Embedding) — preferred for LLMs
$$\text{RoPE}(q, m) = q \cdot e^{im\theta}$$

Applied by rotating query/key vectors by position-dependent angle. Enables length extrapolation.

```python
def apply_rope(x, cos, sin):
    # x: [B, T, H, D], cos/sin: [T, D]
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin
```

---

## Tokenization

| Method | When to use |
|---|---|
| BPE (GPT-2) | General purpose LLMs |
| WordPiece (BERT) | Masked LM pretraining |
| SentencePiece (T5, LLaMA) | Multilingual, subword |
| Character-level | Morphologically rich languages |

---

## Pretraining Objectives

| Objective | Model | Formula |
|---|---|---|
| Causal LM (CLM) | GPT | $\mathcal{L} = -\sum_t \log p(x_t \mid x_{<t})$ |
| Masked LM (MLM) | BERT | $\mathcal{L} = -\sum_{t \in M} \log p(x_t \mid x_{\setminus M})$ |
| Span corruption | T5 | Replace spans with sentinel tokens |
| Contrastive | CLIP, SimCSE | $\mathcal{L}_{\text{InfoNCE}}$ |

---

## Fine-Tuning Strategies

### Full Fine-Tuning
- All parameters updated
- Requires large memory (optimizer states = 2× params for Adam)
- Best accuracy, highest cost

### LoRA (Low-Rank Adaptation)
$$W' = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d,k)$$

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False  # Freeze
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale
```

### QLoRA
- Quantize base model to 4-bit (NF4)
- Train LoRA adapters in 16-bit
- Enables 65B model fine-tuning on single GPU

---

## RLHF Pipeline

```
1. Supervised Fine-Tuning (SFT)
   → Train on human demonstrations
   
2. Reward Model Training
   → Given (prompt, response_A, response_B) + human preference
   → RM outputs scalar reward r(x, y)
   → Loss: -log σ(r(x, y_w) - r(x, y_l))
   
3. PPO / GRPO Optimization
   → Maximize E[r(x, y)] subject to KL(π || π_ref) ≤ δ
   → L_PPO = E[min(ratio * A, clip(ratio, 1±ε) * A)] - β * KL
```

### DPO (Direct Preference Optimization) — simpler alternative
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\!\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

---

## Long Context Techniques

| Technique | Max Context | Tradeoff |
|---|---|---|
| Full attention | ~4K-8K | O(n²) memory |
| Flash Attention 2 | ~32K+ | IO-optimal, same output |
| Sliding window | Unlimited | No global context |
| Mamba (SSM) | Unlimited | O(n) but selective |
| RoPE + YaRN | 128K+ | Interpolation tricks |

---

## RAG (Retrieval-Augmented Generation)

```
Query → Embedding Model → Vector DB (cosine search) → Top-K chunks
     → Prompt: [System] + [Retrieved context] + [Query]
     → LLM generates answer grounded in retrieved docs
```

Key considerations:
- Chunking strategy (fixed-size vs semantic)
- Embedding model alignment with query distribution  
- Re-ranking (cross-encoder) after initial retrieval
- Hybrid search: dense (embedding) + sparse (BM25)
