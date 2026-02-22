# Generative Models — Deep Reference

## Diffusion Models

### Forward Process (fixed Markov chain)
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I)$$

Closed-form sampling at any timestep:
$$q(x_t | x_0) = \mathcal{N}(x_t;\, \sqrt{\bar{\alpha}_t}\,x_0,\, (1-\bar{\alpha}_t)I)$$

where $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$.

### Reverse Process (learned denoiser)
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I)$$

### Training Objective (simplified)
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

```python
class DiffusionTrainer:
    def __init__(self, model, timesteps=1000):
        self.model = model
        self.T = timesteps
        # Cosine noise schedule (preferred over linear)
        s = 0.008
        t = torch.linspace(0, timesteps, timesteps + 1)
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def training_step(self, x0):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        
        sqrt_alpha = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        pred_noise = self.model(x_t, t)
        
        return F.mse_loss(pred_noise, noise)
```

### UNet Backbone (for Diffusion)
- Input: noisy image $x_t$ + timestep embedding $t$
- Architecture: Encoder-decoder with skip connections
- Key: ResBlocks + Self-Attention at low resolutions
- Timestep conditioning: AdaGN (Adaptive Group Norm) or AdaLN

### Classifier-Free Guidance (CFG)
$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

Guidance scale $w > 1$ increases sample quality / reduces diversity tradeoff.

### Flow Matching (modern alternative)
$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

Straight-line interpolation, faster training convergence than DDPM.

---

## GANs

### Objective
$$\min_G \max_D \; \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$$

### Stability Techniques
| Problem | Fix |
|---|---|
| Mode collapse | Minibatch discrimination, Unrolled GAN, WGAN |
| Gradient vanishing for D | Hinge loss, WGAN-GP |
| Training instability | Spectral normalization, EMA of generator weights |
| Low diversity | Truncation trick (at inference), diversity loss |

### WGAN-GP (Wasserstein + Gradient Penalty)
$$\mathcal{L}_D = \mathbb{E}[D(G(z))] - \mathbb{E}[D(x)] + \lambda \mathbb{E}\left[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2\right]$$

```python
def gradient_penalty(D, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interpolated)
    grad = torch.autograd.grad(d_interp, interpolated,
                               grad_outputs=torch.ones_like(d_interp),
                               create_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()
```

---

## VAEs

### ELBO Objective
$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$$

### Reparameterization Trick
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

```python
class VAE(nn.Module):
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss(self, x, x_recon, mu, log_var):
        recon = F.mse_loss(x_recon, x, reduction='sum')
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon + self.beta * kl  # β-VAE: β > 1 for disentanglement
```

---

## Normalizing Flows

Transform simple distribution $p(z)$ to complex $p(x)$ via invertible $f$:
$$\log p(x) = \log p(f^{-1}(x)) + \log\left|\det\frac{\partial f^{-1}}{\partial x}\right|$$

Types: RealNVP (affine coupling), Glow (1×1 conv), Neural Spline Flows (most expressive).

---

## Energy-Based Models

$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)}$$

Training via contrastive divergence or score matching. Inference via MCMC / Langevin dynamics.

---

## Architecture Selection for Generation

```
High-quality images (static)  → Diffusion (DDPM/EDM/Flow Matching)
Real-time generation needed   → GAN (StyleGAN3) or Consistency Models
Disentangled representation   → β-VAE or InfoGAN
Exact likelihood              → Normalizing Flow
Text-conditioned images       → Latent Diffusion (SD) + CLIP conditioning
Video generation              → Diffusion + temporal attention (CogVideo, Sora-like)
Audio generation              → Diffusion (AudioLDM) or autoregressive (VQ-VAE + GPT)
```
