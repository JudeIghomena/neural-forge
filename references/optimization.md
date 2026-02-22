# Optimization & Efficiency — Deep Reference

## Quantization

### Post-Training Quantization (PTQ)

```python
import torch.ao.quantization as quant

# Dynamic quantization (weights only, fastest)
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
)

# Static quantization (weights + activations)
model.qconfig = quant.get_default_qconfig('x86')
quant.prepare(model, inplace=True)
# Run calibration data through model
calibrate(model, calibration_loader)
quant.convert(model, inplace=True)
```

### QAT (Quantization-Aware Training)
```python
model.train()
model.qconfig = quant.get_default_qat_qconfig('x86')
quant.prepare_qat(model, inplace=True)
# Train with fake quantization nodes inserted
train(model, ...)
model.eval()
quant.convert(model, inplace=True)
```

### GPTQ (LLM Quantization)
- Weight-only, 4-bit, minimal accuracy loss
- Reconstructs each layer's weights to minimize output error
- `auto-gptq` or `bitsandbytes` library

### Key Insight
$$W_q = \text{round}\!\left(\frac{W}{\Delta}\right), \quad \Delta = \frac{\max|W|}{2^{b-1}-1}$$

INT8 → 4× smaller, 2-4× faster on modern hardware (especially edge/mobile).

---

## Pruning

### Unstructured Pruning (Magnitude)
```python
import torch.nn.utils.prune as prune

# Prune 50% of weights by magnitude (globally)
parameters_to_prune = [(module, 'weight') for module in model.modules()
                        if isinstance(module, nn.Linear)]
prune.global_unstructured(parameters_to_prune,
                           pruning_method=prune.L1Unstructured, amount=0.5)
```

### Structured Pruning (Channel/Head)
```python
# Remove attention heads with lowest importance scores
def prune_attention_heads(model, heads_to_prune):
    for layer_idx, heads in heads_to_prune.items():
        model.transformer.h[layer_idx].attn.prune_heads(heads)
```

### Lottery Ticket Hypothesis
1. Train to convergence → find sparse mask (top-k weights by magnitude)
2. Reset remaining weights to *initial* values
3. Retrain with sparse mask from scratch
4. Sparse subnetwork ("winning ticket") matches dense accuracy

---

## Knowledge Distillation

$$\mathcal{L}_{\text{KD}} = (1-\alpha)\mathcal{L}_{\text{CE}}(y, \sigma(z_s)) + \alpha T^2 \mathcal{L}_{\text{CE}}(\sigma(z_t/T), \sigma(z_s/T))$$

where $T$ is temperature (higher = softer targets), $\alpha$ balances hard/soft loss.

```python
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    soft_targets = F.softmax(teacher_logits / T, dim=-1)
    soft_pred = F.log_softmax(student_logits / T, dim=-1)
    soft_loss = F.kl_div(soft_pred, soft_targets, reduction='batchmean') * T**2
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### Feature Distillation (Intermediate layers)
$$\mathcal{L}_{\text{feat}} = \sum_l \|f_s^l - g(f_t^l)\|_F^2$$

where $g$ is a projection (to handle dimension mismatch).

---

## PEFT (Parameter-Efficient Fine-Tuning)

| Method | Params | Memory | Quality |
|---|---|---|---|
| Full FT | 100% | High | Best |
| LoRA | ~0.1-1% | Low | Near-full |
| Prefix Tuning | ~0.1% | Low | Good |
| Adapter | ~1-3% | Medium | Good |
| Prompt Tuning | <0.01% | Lowest | Moderate |

### LoRA Rank Selection
- $r = 4$: tiny models, fast experiments
- $r = 8-16$: standard fine-tuning
- $r = 64$: domain adaptation, large shifts
- Target modules: Q, V projections (standard); add K, O, FFN for larger tasks

### DoRA (Weight-Decomposition LoRA)
$$W = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|}$$

Decomposes into magnitude ($m$) and direction ($\frac{W}{\|W\|}$), further improves LoRA.

---

## Neural Architecture Search (NAS)

### DARTS (Differentiable NAS)
$$\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o'}\exp(\alpha_{o'}^{(i,j)})} \cdot o(x)$$

Jointly optimize architecture $\alpha$ and weights $w$ via bi-level optimization.

### One-Shot NAS (weight sharing)
Train a **supernet** that contains all candidate architectures as subgraphs.
Each forward pass samples a random subarchitecture. Then evaluate subnets on validation.

---

## torch.compile() (PyTorch 2.0+)

```python
# One line — massive speedups (up to 2× on GPUs)
model = torch.compile(model)

# Options
model = torch.compile(model, mode='reduce-overhead')   # Best for training
model = torch.compile(model, mode='max-autotune')       # Best for inference
model = torch.compile(model, dynamic=True)              # Variable-length sequences
```

Internals: TorchDynamo (graph capture) → TorchInductor (Triton kernel codegen)

---

## Memory Optimization

```python
# 1. Gradient checkpointing (trade compute for memory)
from torch.utils.checkpoint import checkpoint_sequential
output = checkpoint_sequential(model.layers, segments=4, input=x)

# 2. Activation offloading to CPU
# 3. ZeRO (DeepSpeed) — shard optimizer states, gradients, params
import deepspeed
model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config)

# 4. FlashAttention (IO-aware, no full attention matrix materialization)
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v, causal=True)
```

### Memory Breakdown (per parameter)
| Component | FP32 | Mixed (BF16) |
|---|---|---|
| Parameters | 4 bytes | 2 bytes |
| Gradients | 4 bytes | 2 bytes |
| Adam states | 8 bytes | 4 bytes (master weights) |
| **Total** | **16 bytes** | **~8 bytes** |

A 7B parameter model needs ~112GB (FP32) or ~56GB (BF16 + AMP).

---

## Profiling Tools

```python
# PyTorch Profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True, with_stack=True
) as prof:
    model(x)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# Memory profiler
torch.cuda.memory_summary()
torch.cuda.max_memory_allocated() / 1e9  # GB
```
