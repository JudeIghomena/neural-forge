# ⚡ Neural Forge

> **Advanced Deep Neural Network Engineering Skill for Claude**  
> A full-spectrum DNN co-pilot covering every major domain of modern deep learning.

[![Branch](https://img.shields.io/badge/branch-skills-blue)](https://github.com/JudeIghomena/neural-forge/tree/skills)
[![Lines](https://img.shields.io/badge/knowledge-1984%20lines-brightgreen)](#)
[![Domains](https://img.shields.io/badge/domains-10-orange)](#domains)
[![License](https://img.shields.io/badge/license-MIT-purple)](LICENSE)

---

## What is Neural Forge?

Neural Forge is a Claude skill that transforms Claude into a world-class deep learning engineering partner. It covers architecture design, training pipelines, debugging, research synthesis, and model optimization — across **all domains** of deep learning, **all output formats**, and **all expertise levels**.

---

## Capability Matrix

| Capability | Description |
|---|---|
| 🏗️ **Architecture Design** | Novel architectures, hybrid models, custom layers, attention variants |
| 🔧 **Training Pipelines** | Distributed training, mixed precision, gradient accumulation, curriculum learning |
| 🐛 **Debugging** | Vanishing/exploding gradients, loss spikes, mode collapse, NaN debugging |
| 📄 **Research Synthesis** | Paper → implementation blueprint, SOTA comparisons |
| ⚡ **Optimization** | Pruning, quantization (INT8/FP16/BF16), NAS, distillation, PEFT/LoRA |
| 📐 **Math Formulations** | LaTeX derivations, loss function design, theoretical analysis |
| 📊 **Visualization** | Architecture diagrams (Mermaid/ASCII), training curves, attention maps |

---

## Domains

| Domain | Reference File | Coverage |
|---|---|---|
| 👁️ Computer Vision | `references/cv.md` | CNNs, ViTs, MAE, DINO, detection, segmentation |
| 🧠 NLP & LLMs | `references/nlp_llm.md` | Transformers, RoPE, LoRA, RLHF, DPO, RAG |
| 🎮 Reinforcement Learning | `references/rl.md` | PPO, SAC, DQN, GRPO, multi-agent |
| 🎨 Generative Models | `references/generative.md` | Diffusion, GANs, VAEs, Flow Matching |
| 🕸️ Graph Neural Networks | `references/gnn.md` | GCN, GAT, GIN, GraphSAGE, temporal graphs |
| 🔬 Scientific ML | `references/scientific_ml.md` | PINNs, FNO, Neural ODEs, equivariant nets, AlphaFold |
| 📈 Time-Series | `references/timeseries.md` | LSTM, TCN, Mamba, TFT, PatchTST, anomaly detection |
| ⚙️ Optimization & Efficiency | `references/optimization.md` | Quantization, pruning, distillation, LoRA, NAS |
| 🏭 Training Infrastructure | `references/training_infra.md` | DDP, FSDP, mixed precision, profiling, debugging |
| 🌐 Multimodal | `references/multimodal.md` | CLIP, VLMs, Q-Former, video, hallucination mitigation |

---

## Output Formats

Neural Forge adapts its output to the request:

- **Runnable code** — PyTorch (primary), JAX, TensorFlow equivalents
- **Mathematical formulations** — LaTeX equations for every key operation
- **Architecture diagrams** — ASCII and Mermaid diagrams
- **Research pseudocode** — clean, implementation-ready blueprints

---

## Structure

```
neural-forge/
├── SKILL.md                    # Master skill — routing logic, universal protocols
└── references/
    ├── cv.md                   # Computer Vision
    ├── nlp_llm.md              # NLP & Large Language Models
    ├── rl.md                   # Reinforcement Learning
    ├── generative.md           # Generative Models
    ├── gnn.md                  # Graph Neural Networks
    ├── scientific_ml.md        # Scientific Machine Learning
    ├── timeseries.md           # Time-Series & Sequential Models
    ├── optimization.md         # Model Optimization & Efficiency
    ├── training_infra.md       # Training Infrastructure
    └── multimodal.md           # Multimodal Systems
```

---

## How It Works

The skill uses **progressive disclosure** — Claude loads only what's needed:

1. `SKILL.md` triggers on any DNN-related query and handles routing
2. The relevant domain reference file is read for deep dives
3. Output format is auto-selected based on query type and user expertise

---

## Example Queries

```
"Design a hybrid CNN-Transformer for medical image segmentation"
"Implement PPO from scratch in PyTorch with GAE"
"My transformer loss spikes at step 3000, how do I debug this?"
"Explain the math behind DDPM and give me a minimal implementation"
"How does LoRA work and when should I use DoRA instead?"
"Build a PINN to solve the Navier-Stokes equations"
"What's the difference between GCN, GAT, and GIN?"
"Implement Flash Attention 2 with causal masking"
```

---

## Author

**Jude Ighomena**  
Award-Winning AI & Telecom Infrastructure Leader  
Co-Founder, iRaven Group UK 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Jude%20Ighomena-blue?logo=linkedin)](https://linkedin.com/in/jude-ighomena)

---

## License

MIT © 2026 Jude Ighomena — see [LICENSE](LICENSE)
