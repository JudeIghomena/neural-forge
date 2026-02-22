# Computer Vision — Deep Reference

## Architecture Taxonomy

### CNN Lineage
```
LeNet → AlexNet → VGG → ResNet → DenseNet → EfficientNet → ConvNeXt
                                ↓
                    ResNeXt (grouped convolutions)
                    SENet (channel attention)
                    CBAM (spatial + channel attention)
```

### Vision Transformer Lineage
```
ViT → DeiT (distillation) → Swin (hierarchical, shifted window)
    → BEiT (masked image modeling)
    → MAE (masked autoencoder)
    → DINOv2 (self-supervised)
    → CoAtNet (conv + attention hybrid)
```

---

## ResNet Block (PyTorch)

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        return self.act(out)
```

---

## Vision Transformer (ViT)

### Mathematical Formulation

Patch embedding: given image $x \in \mathbb{R}^{H \times W \times C}$, split into $N = \frac{HW}{P^2}$ patches:

$$z_0 = [x_{cls}; x_p^1 E; x_p^2 E; \ldots; x_p^N E] + E_{pos}$$

where $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the patch embedding matrix.

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_embed
```

---

## Object Detection

### Architecture Choices
- **One-stage**: YOLO (speed-first), FCOS (anchor-free), DETR (transformer-based, set prediction)
- **Two-stage**: Faster R-CNN (accuracy-first), Cascade R-CNN

### DETR Loss (Hungarian matching)
$$\mathcal{L}_{\text{DETR}} = \sum_{i=1}^{N} \left[ -\log p_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{c_i \neq \emptyset} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right]$$

---

## Semantic Segmentation

```python
# UNet-style decoder
class SegDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes):
        super().__init__()
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(enc, enc//2, 2, stride=2),
                ResidualBlock(enc//2 + skip, enc//2)
            )
            for enc, skip in zip(encoder_channels[:-1], encoder_channels[1:])
        ])
        self.head = nn.Conv2d(encoder_channels[-1]//2, num_classes, 1)
```

---

## Key Tricks for CV

1. **Augmentation**: RandAugment, MixUp, CutMix, Copy-Paste (detection)
2. **Normalization**: BatchNorm for large batches, LayerNorm/GroupNorm for small
3. **Initialization**: He (ReLU), Xavier (Sigmoid/Tanh), truncated normal for ViTs
4. **Regularization**: Stochastic depth (DropPath), label smoothing
5. **Multi-scale training**: Random resize during training, test-time augmentation

---

## Self-Supervised Pretraining

### MAE (Masked Autoencoder)
- Mask 75% of patches randomly
- Encode only visible patches (efficient)
- Decode from masked tokens → reconstruct pixel values
- Loss: MSE on masked patches only

$$\mathcal{L}_{\text{MAE}} = \frac{1}{|\Omega_M|} \sum_{p \in \Omega_M} \|x_p - \hat{x}_p\|^2$$

### DINO / DINOv2
- Self-distillation: student network matches teacher (EMA of student)
- Centering + sharpening to prevent collapse
- No negative pairs needed
