# Reinforcement Learning — Deep Reference

## Core Framework

**MDP**: $(S, A, P, R, \gamma)$ — States, Actions, Transition, Reward, Discount

**Bellman Equations:**
$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s\right]$$
$$Q^\pi(s,a) = R(s,a) + \gamma \mathbb{E}_{s'}[V^\pi(s')]$$

---

## Algorithm Taxonomy

```
Model-Free RL
├── Value-Based
│   ├── DQN (discrete actions)
│   ├── Double DQN (reduce overestimation)
│   ├── Dueling DQN (V + A decomposition)
│   └── Rainbow (all tricks combined)
├── Policy Gradient
│   ├── REINFORCE (high variance, baseline helps)
│   ├── A2C / A3C (actor-critic)
│   ├── PPO (clipped surrogate, most practical)
│   ├── TRPO (trust region, precursor to PPO)
│   └── SAC (entropy-regularized, continuous)
└── Actor-Critic
    ├── TD3 (deterministic, twin critics)
    └── SAC (stochastic, max entropy)

Model-Based RL
├── Dyna (learn model, plan with it)
├── MuZero (learn model + planning via MCTS)
└── Dreamer (latent world model)
```

---

## PPO (Proximal Policy Optimization)

Most practical algorithm. Use this as default for most tasks.

### Objective
$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ is the probability ratio.

### Full Loss
$$\mathcal{L} = -\mathcal{L}^{\text{CLIP}} + c_1 \mathcal{L}^{\text{VF}} - c_2 H[\pi_\theta]$$

```python
class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        # Shared backbone (optional)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor = nn.Linear(hidden, act_dim)   # Policy head
        self.critic = nn.Linear(hidden, 1)         # Value head

    def get_action_and_value(self, obs):
        h = self.backbone(obs)
        logits = self.actor(h)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(h)

def ppo_update(agent, optimizer, batch, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    obs, actions, old_logprobs, advantages, returns = batch
    
    _, new_logprobs, entropy, values = agent.get_action_and_value(obs)
    
    ratio = (new_logprobs - old_logprobs).exp()
    pg_loss = -torch.min(
        ratio * advantages,
        ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    ).mean()
    
    vf_loss = F.mse_loss(values.squeeze(), returns)
    loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy.mean()
    
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
    optimizer.step()
```

---

## SAC (Soft Actor-Critic) — Continuous Actions

### Maximum Entropy Objective
$$J(\pi) = \sum_t \mathbb{E}\left[r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$$

Key features:
- Twin Q-networks (prevents overestimation)
- Auto-tuned temperature $\alpha$
- Off-policy (replay buffer)
- Best default for continuous control

---

## DQN (Discrete Actions)

```python
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    
    def forward(self, x):
        return self.net(x)

# Double DQN update
def dqn_loss(online, target, batch, gamma=0.99):
    obs, actions, rewards, next_obs, dones = batch
    with torch.no_grad():
        # Double DQN: online selects action, target evaluates
        next_actions = online(next_obs).argmax(dim=1)
        next_q = target(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q = rewards + gamma * next_q * (1 - dones)
    current_q = online(obs).gather(1, actions.unsqueeze(1)).squeeze()
    return F.smooth_l1_loss(current_q, target_q)  # Huber loss for stability
```

---

## Advantage Estimation

### GAE (Generalized Advantage Estimation)
$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

- $\lambda = 0$: TD(0), low variance, high bias
- $\lambda = 1$: Monte Carlo, high variance, zero bias
- $\lambda = 0.95$: standard PPO default

---

## RLHF Architecture

```
Pretraining → SFT → Reward Model → PPO/GRPO
                                  ↓
                     π_θ maximizes r_φ(x,y)
                     subject to KL(π_θ || π_ref) ≤ δ
```

### GRPO (Group Relative Policy Optimization) — DeepSeek approach
- Sample $G$ outputs per prompt
- Reward each, normalize within group
- No value function needed → memory efficient

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\sum_{i=1}^G \frac{\pi_\theta(o_i|q)}{\pi_{\theta_\text{old}}(o_i|q)} \hat{A}_i - \beta \text{KL}(\pi_\theta || \pi_{\text{ref}})\right]$$

---

## Multi-Agent RL

- **Cooperative**: QMIX, MAPPO (centralized training, decentralized execution)
- **Competitive**: MARL with self-play (AlphaGo, OpenAI Five)
- **Communication**: CommNet, TarMAC (learned agent messaging)

---

## Algorithm Selection Guide

| Task | Recommended Algorithm |
|---|---|
| Discrete actions, simple env | DQN / Double DQN |
| Continuous control (MuJoCo) | SAC or TD3 |
| On-policy, general purpose | PPO |
| Sparse rewards | Curiosity-driven (ICM) + PPO |
| Large action spaces | Action Embedding + DQN |
| Multi-task / meta-RL | MAML, RL² |
| RLHF for LLMs | PPO or GRPO + KL penalty |
