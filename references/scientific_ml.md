# Scientific ML — Deep Reference

## Physics-Informed Neural Networks (PINNs)

### Core Idea
Encode the PDE as a loss term — the network learns to satisfy both data AND the governing physics.

### Loss Function
$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{data}}}_{\text{observations}} + \lambda_r\underbrace{\mathcal{L}_{\text{PDE}}}_{\text{residual}} + \lambda_b\underbrace{\mathcal{L}_{\text{BC}}}_{\text{boundary conds.}} + \lambda_i\underbrace{\mathcal{L}_{\text{IC}}}_{\text{initial conds.}}$$

Example: 2D Navier-Stokes

$$\mathcal{L}_r = \left\|u_t + u u_x + v u_y + p_x - \nu\nabla^2 u\right\|^2 + \left\|u_x + v_y\right\|^2$$

```python
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 1]):
        super().__init__()
        # Tanh activation preferred for smooth PDE solutions
        self.net = nn.Sequential(*[
            nn.Sequential(nn.Linear(layers[i], layers[i+1]), nn.Tanh())
            for i in range(len(layers)-2)
        ] + [nn.Linear(layers[-2], layers[-1])])

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

def pde_residual(model, x, t, nu=0.01):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    # Automatic differentiation for PDE terms
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    # Burgers equation: u_t + u*u_x = nu * u_xx
    return u_t + u * u_x - nu * u_xx
```

### Failure Modes & Fixes
| Problem | Fix |
|---|---|
| Stiff PDEs (multiple timescales) | Adaptive loss weighting (NTK-based) |
| High-frequency features | Fourier feature embeddings |
| Sharp gradients | Adaptive sampling (RAR, RAD) |
| Large domains | Decomposition (XPINNs, FBPINNs) |

---

## Neural Operators

Learn mappings between function spaces — generalize across different discretizations.

### Fourier Neural Operator (FNO)
$$(\mathcal{F} v)(k) = \int v(x) e^{-2\pi i \langle x,k\rangle} dx$$

FNO layer: $v_{t+1}(x) = \sigma(W v_t(x) + \mathcal{F}^{-1}(R_\phi \cdot \mathcal{F}(v_t))(x))$

```python
import torch.fft as fft

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        x_ft = fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            'bixy,ioxy->boxy', x_ft[:, :, :self.modes1, :self.modes2], self.weights
        )
        return fft.irfft2(out_ft, s=x.shape[-2:])
```

### DeepONet (Deep Operator Network)
$$G_\theta(u)(y) = \sum_{k=1}^p b_k(y) \cdot t_k(u(x_1),\ldots,u(x_m))$$

Branch net encodes input function; trunk net encodes query location.

---

## Neural ODEs

Replace discrete ResNet layers with continuous ODE:

$$\frac{dh(t)}{dt} = f_\theta(h(t), t), \quad h(0) = x$$

```python
from torchdiffeq import odeint_adjoint

class NeuralODE(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.odefunc = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, dim)
        )
        self.integration_time = torch.tensor([0.0, 1.0])

    def forward(self, x):
        return odeint_adjoint(
            self.odefunc, x, self.integration_time,
            method='dopri5', rtol=1e-3, atol=1e-3
        )[-1]  # Return final state
```

Adjoint method: $O(1)$ memory backprop regardless of depth.

---

## Equivariant Networks

For physical systems with symmetries (rotations, translations, reflections).

### Key Principle
$$f(T_g x) = T_g' f(x) \quad \forall g \in G$$

If $T_g' = T_g$: **equivariant**. If $T_g' = I$: **invariant**.

| Application | Symmetry Group | Architecture |
|---|---|---|
| Molecular energy | SE(3) / E(3) | EGNN, NequIP, PaiNN |
| Protein structure | SE(3) | AlphaFold2 (Evoformer + IPA) |
| Point clouds | SO(3) | Vector Neurons, EQNN |
| Crystal structures | Space group | MatFormer |

```python
# EGNN layer (E(3)-equivariant)
class EGNNLayer(nn.Module):
    def forward(self, h, x, edge_index):
        i, j = edge_index
        diffs = x[i] - x[j]  # Relative positions (equivariant)
        dists = diffs.norm(dim=-1, keepdim=True)
        msg = self.message_mlp(torch.cat([h[i], h[j], dists], dim=-1))
        # Update coordinates equivariantly
        x_agg = scatter(diffs * self.coord_mlp(msg), i, dim=0)
        h_new = self.node_mlp(torch.cat([h, scatter(msg, i, dim=0)], dim=-1))
        return h_new, x + x_agg
```

---

## Protein Structure (AlphaFold2 concepts)

- **Evoformer**: Transformer operating on MSA (multiple sequence alignment) + pair representations
- **IPA (Invariant Point Attention)**: SE(3)-equivariant attention for 3D backbone
- **Recycling**: iterative refinement (run model 3× feeding output back as input)
- **FAPE loss**: Frame-Aligned Point Error (backbone atom deviation in local frames)

$$\mathcal{L}_{\text{FAPE}} = \frac{1}{NT}\sum_{i,j} \text{clip}\left(\left\|T_i^{-1} x_j - \hat{T}_i^{-1} \hat{x}_j\right\|, 10\text{Å}\right)$$

---

## Scientific ML Best Practices

1. **Non-dimensionalize** inputs — normalize to $[0,1]$ or $[-1,1]$ for numerical stability
2. **Tanh activations** for PINNs (smooth, bounded gradients for PDE solutions)
3. **Curriculum in time** — train on early time first, extend progressively
4. **Adaptive sampling** — oversample high-residual regions
5. **Double precision** — use `torch.float64` for stiff PDEs
6. **Physics-consistency at inference** — project outputs to satisfy conservation laws
