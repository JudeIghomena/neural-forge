# Multimodal Models — Deep Reference

## Contrastive Multimodal (CLIP family)

### CLIP Training Objective
Given a batch of (image, text) pairs, maximize cosine similarity of matched pairs:

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^N\left[\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}} + \log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}}\right]$$

where $s_{ij} = \frac{f_I(x_i)^\top f_T(y_j)}{\|f_I(x_i)\|\|f_T(y_j)\|}$ and $\tau$ is a learned temperature.

```python
class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(temperature))

    def forward(self, image_feats, text_feats):
        # Normalize
        image_feats = F.normalize(image_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        
        logits = (image_feats @ text_feats.T) / self.temp.exp()
        labels = torch.arange(len(logits), device=logits.device)
        
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return (loss_i + loss_t) / 2
```

### CLIP Variants
| Model | Improvement | Notes |
|---|---|---|
| CLIP | Baseline | 400M image-text pairs |
| OpenCLIP | Open reproduction | Better scaling laws |
| SigLIP | Sigmoid loss (no softmax across batch) | Better at small batch sizes |
| DFN-CLIP | Data filtering | Quality > quantity |
| EVA-CLIP | Scaled ViT encoder | Best zero-shot |

---

## Vision-Language Models (VLMs)

### Architecture Pattern
```
Image → Vision Encoder (CLIP/ViT) → [Connector] → LLM
Text  ────────────────────────────────────────────────↗
```

### Connector Types

**Linear Projection (LLaVA-1.0)**:
```python
self.mm_projector = nn.Linear(vision_hidden, llm_hidden)
```

**MLP Projection (LLaVA-1.5)** — significant improvement:
```python
self.mm_projector = nn.Sequential(
    nn.Linear(vision_hidden, llm_hidden),
    nn.GELU(),
    nn.Linear(llm_hidden, llm_hidden)
)
```

**Q-Former (BLIP-2)** — cross-attention queries:
```python
# Learned query tokens attend to image features
# Outputs fixed N=32 tokens regardless of image resolution
class QFormer(nn.Module):
    def __init__(self, num_queries=32, hidden=768, n_heads=12, n_layers=12):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, hidden))
        self.layers = nn.ModuleList([QFormerLayer(hidden, n_heads) for _ in range(n_layers)])
    
    def forward(self, image_features):
        B = image_features.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, image_features)  # Cross-attention to image
        return queries
```

**Perceiver Resampler (Flamingo)** — similar to Q-Former, more scalable.

---

## Training Stages for VLMs

### Stage 1: Vision-Language Alignment
- Freeze: LLM + Vision Encoder
- Train: Connector only
- Data: Image-caption pairs (LAION, CC3M, etc.)
- Goal: Map visual tokens into LLM's embedding space
- Duration: Short (~1 epoch on 600K pairs)

### Stage 2: Visual Instruction Tuning
- Freeze: Vision Encoder
- Train: Connector + LLM (or LoRA on LLM)
- Data: Visual instruction following data (LLaVA-mix, ShareGPT4V)
- Goal: Follow multimodal instructions
- Duration: 1-2 epochs

### Stage 3 (Optional): RLHF / DPO
- Align VLM to human preferences
- Reduce hallucination, improve helpfulness

---

## Audio-Language Models

### Architecture
```
Audio → Encoder (Whisper / wav2vec2 / HuBERT) → Projection → LLM
```

Key challenge: audio has much higher token density than text/image.

Strategies:
- **Pooling**: downsample audio features (2-4× stride)
- **Q-Former**: compress to fixed tokens
- **Whisper as feature extractor**: pre-trained speech understanding

---

## Video-Language Models

### Temporal Modeling Approaches

**Frame sampling** (simplest):
- Sample K frames at fixed intervals → treat as K images
- Works surprisingly well for short videos

**Temporal attention**:
- Add temporal transformer after spatial ViT
- Space-Time Attention: $\text{Attn}(Q, [K_{\text{spatial}}; K_{\text{temporal}}], [V_s; V_t])$

**Video tokens**:
```python
# Video: [B, T, C, H, W] → patch embed each frame → [B, T*N_patches, D]
# Then apply temporal position encoding
temporal_pos = nn.Parameter(torch.zeros(1, T, 1, D))
spatial_pos = nn.Parameter(torch.zeros(1, 1, N_patches, D))
embeddings = patch_embeds + temporal_pos + spatial_pos
```

---

## Hallucination Mitigation

VLMs frequently hallucinate objects not in the image.

Causes:
1. Language priors dominating visual signal
2. Insufficient visual grounding during training
3. Decoding strategy (beam search amplifies early errors)

Fixes:
- **Visual contrastive decoding**: compare logits with/without image
  $$p_{\text{VCD}}(y|v,t) \propto \log p(y|v,t) - \log p(y|t)$$
- **RLHF / DPO** on preference data with hallucination labels
- **Grounding supervision**: train with region-level descriptions
- **OPERA**: overconfidence penalty in beam search

---

## Cross-Modal Fusion Strategies

| Strategy | Description | Best for |
|---|---|---|
| Early fusion | Concatenate raw features | Simple tasks |
| Mid fusion | Fuse at intermediate layers | Most VLMs |
| Late fusion | Fuse final predictions | Ensemble-style |
| Cross-attention | Modality A queries modality B | Flamingo, BLIP-2 |
| Shared encoder | Single encoder processes all modalities | ImageBind |
