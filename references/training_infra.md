# Training Infrastructure — Deep Reference

## Distributed Training Strategies

```
DDP (Distributed Data Parallel)      — Replicate model, shard data. Default for multi-GPU.
FSDP (Fully Sharded Data Parallel)   — Shard model + gradients + optimizer. For large models.
Tensor Parallel                       — Shard individual layers across GPUs (Megatron-LM style).
Pipeline Parallel                     — Different layers on different GPUs.
Sequence Parallel                     — Shard sequence dimension (for long-context LLMs).
```

### DDP (Most Common)
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group('nccl')
model = DDP(model.cuda(), device_ids=[local_rank])
# Launch: torchrun --nproc_per_node=8 train.py
```

### FSDP (Large Models)
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                   reduce_dtype=torch.float32),
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
)
```

---

## Mixed Precision Training

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        # BF16 preferred: wider range, no NaN overflow
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

**BF16 vs FP16**:
- BF16: same exponent range as FP32 (8 bits), fewer mantissa bits → no overflow
- FP16: more mantissa bits, smaller range → requires loss scaling

---

## Optimizer Selection

| Optimizer | When to use | Key hyperparams |
|---|---|---|
| AdamW | Default for most DL | lr, β₁=0.9, β₂=0.999, wd=0.01-0.1 |
| Adam + Warmup | Transformers | Warmup steps = 4% of total |
| SGD + momentum | CV (often beats Adam with tuning) | lr, momentum=0.9, wd=1e-4 |
| Lion | LLMs, memory-efficient | lr ~3-10× smaller than Adam |
| Muon | LLMs (emerging, fastest convergence) | — |
| Sophia | LLMs, second-order | hessian_update_freq |

### Learning Rate Schedule

```python
import math

def cosine_with_warmup(step, warmup_steps, total_steps, min_lr=0.1):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + (1 - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda s: cosine_with_warmup(s, warmup_steps=1000, total_steps=100000)
)
```

---

## Gradient Debugging

```python
# Check for NaN/Inf gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            print(f"BAD GRADIENT: {name}")

# Gradient norm monitoring
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
print(f"Gradient norm: {total_norm:.4f}")

# Gradient flow visualization
def plot_grad_flow(named_params):
    ave_grads = []
    layers = []
    for n, p in named_params:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    return layers, ave_grads

# Register hooks for gradient monitoring
hooks = []
for name, module in model.named_modules():
    hook = module.register_backward_hook(
        lambda m, gi, go, n=name: print(f"{n}: grad_out norm = {go[0].norm():.4f}" if go[0] is not None else "None")
    )
    hooks.append(hook)
```

---

## Experiment Tracking

```python
import wandb

wandb.init(project='dnn-experiment', config={
    'model': 'transformer-base',
    'lr': 3e-4,
    'batch_size': 256,
    'epochs': 100,
})

# Log during training
wandb.log({
    'loss/train': train_loss,
    'loss/val': val_loss,
    'lr': scheduler.get_last_lr()[0],
    'grad_norm': total_norm,
    'epoch': epoch,
})

# Log model checkpoint
wandb.save('checkpoint.pt')
```

---

## Data Pipeline Optimization

```python
from torch.utils.data import DataLoader

# Optimal DataLoader config
loader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=8,           # 4-8× GPU count
    pin_memory=True,         # Faster host→device transfer
    prefetch_factor=2,       # Prefetch batches per worker
    persistent_workers=True, # Don't kill workers between epochs
    drop_last=True,          # Avoid partial batches (important for BN)
)

# Async prefetcher class for maximum GPU utilization
class Prefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
    
    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = {k: v.cuda(non_blocking=True) 
                               for k, v in self.next_batch.items()}
    
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch
```

---

## Checkpoint Best Practices

```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'loss': best_loss,
    'config': config,
}, 'checkpoint.pt')

# Load (always map to right device)
checkpoint = torch.load('checkpoint.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
# Transfer to GPU after loading on CPU (avoid OOM)
model = model.to(device)
```

---

## Common Training Bugs Checklist

- [ ] `.zero_grad()` called before `.backward()`?
- [ ] Model in `.train()` mode for training, `.eval()` for validation?
- [ ] `torch.no_grad()` during validation?
- [ ] Loss normalized by batch size (or using `reduction='mean'`)?
- [ ] Random seeds set for reproducibility?
- [ ] Data shuffled only during training, not validation?
- [ ] Learning rate warmup for transformers?
- [ ] Gradient clipping enabled (especially for RNNs/Transformers)?
- [ ] EMA of model weights for stable evaluation?
- [ ] Deterministic CUDA ops disabled for debugging, enabled for production?
