
# 🌀 Theory of AEON-NN (Adaptive Evolutionary Online Neural Network)

## 1. Core Principle

AEON is **not an optimizer added to a neural network**.
It *is the neural network itself*, whose parameters evolve according to **adaptive evolutionary laws**.

Instead of separating:

* **Model (θ)** and
* **Optimizer (updates)**

AEON fuses them: the *laws of parameter evolution are the network’s own dynamics*.

---

## 2. Evolutionary Law

Each parameter (weight or bias) evolves as:

$$
\theta_{t+1} = \theta_t - \alpha \, \nabla_\theta L + \mu \, \Delta\theta_{t} + \sigma \, \xi_t
$$

Where:

* **α** = learning gain (step size)
* **μ** = momentum factor
* **σξ** = adaptive noise (exploration / dither)
* **Δθₜ** = previous update (memory)
* **L** = loss function

This is both **differentiable** (gradients flow through loss) and **evolutionary** (stochastic noise drives exploration).

---

## 3. Online Adaptation

Unlike offline optimizers (SGD, Adam) where α, μ, σ are fixed or scheduled externally:

* In AEON, these coefficients **adapt in real time** based on:

  * Loss changes (ΔL)
  * Gradient norms (‖g‖)
  * Stability vs. variability (EMA observers)

Thus, training is not “one optimization run” → it is a **continual, online process**.

---

## 4. Implementation in Code

* Each layer (`AEONLinear`) is **self-evolving**: it updates its weights directly with gradients, momentum, and noise.
* The network (`AEONRegressor`) is just a stack of these AEON layers.
* Training loop = forward pass → loss → `model.evolve(loss)` instead of `optimizer.step()`.

This collapses the optimizer into the network itself.

---

## 5. Demonstration: Sine Regression

We tested AEON on a **sine wave fitting problem**:

* Input: $x \in [-2\pi, 2\pi]$
* Target: $y = \sin(x)$
* Model: 3-layer AEON-NN with tanh activations.
* Behavior:

  * Early epochs: network wiggles, explores.
  * Mid epochs: converges toward sine structure.
  * Late epochs: oscillates slightly due to residual noise (unless annealed).

This shows AEON can **learn continuously** and **self-adapt**, without a standard optimizer.

---

## 6. Theoretical Interpretation

* **AEON = Neural Network + Evolutionary Law**
  Not optimizer + network, but one entity.

* **Markovian**: Next state depends only on current parameters, gradients, and coefficients.

* **Differentiable**: Evolution step is fully compatible with autograd.

* **Evolutionary**: Noise injection ensures continual exploration, avoiding premature convergence.

* **Online**: No external schedules, always adapting in real time.

---

✅ In short:
**AEON-NN is a self-evolving, online neural network that merges the ideas of optimizer, dynamics, and architecture into one differentiable evolutionary system.**

---

### PSEUDOCODES

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# AEON Linear Layer (Stabilized)
# ---------------------------
class AEONLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 alpha=0.01, sigma=0.01, momentum=0.9):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.alpha = alpha
        self.sigma = sigma
        self.momentum = momentum
        self.prev_update = torch.zeros_like(self.weight)

        # adaptive gains
        self.alpha_gain = 1.0
        self.sigma_gain = 1.0

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def evolve(self, loss, dL_signal):
        grads = torch.autograd.grad(
            loss, [self.weight, self.bias] if self.bias is not None else [self.weight],
            retain_graph=True, allow_unused=True
        )
        grad_w = grads[0]
        grad_b = grads[1] if self.bias is not None else None

        # clamp adaptive gains
        self.alpha_gain = float(min(max(self.alpha_gain, 0.5), 2.0))
        self.sigma_gain = float(min(max(self.sigma_gain, 0.5), 2.0))

        # adapt based on global signal
        if dL_signal < 0:   # improvement
            self.alpha_gain *= 1.01
            self.sigma_gain *= 0.99
        else:               # plateau or worse
            self.alpha_gain *= 0.99
            self.sigma_gain *= 0.99

        # effective α, σ
        alpha_eff = self.alpha * self.alpha_gain
        sigma_eff = self.sigma * self.sigma_gain

        # hybrid update: mostly gradient, small noise
        noise_w = torch.randn_like(self.weight) * sigma_eff
        update_w = -alpha_eff * grad_w + self.momentum * self.prev_update + 0.1 * noise_w

        with torch.no_grad():
            self.weight.add_(update_w)
            if self.bias is not None and grad_b is not None:
                noise_b = torch.randn_like(self.bias) * sigma_eff
                self.bias.add_(-alpha_eff * grad_b + 0.1 * noise_b)

        self.prev_update = update_w.detach()

# ---------------------------
# AEON Network for Regression
# ---------------------------
class AEONRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = AEONLinear(1, 32)
        self.fc2 = AEONLinear(32, 32)
        self.fc3 = AEONLinear(32, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

    def evolve(self, loss, dL_signal):
        self.fc1.evolve(loss, dL_signal)
        self.fc2.evolve(loss, dL_signal)
        self.fc3.evolve(loss, dL_signal)

# ---------------------------
# Training Loop with Plot
# ---------------------------
def train_aeon_sine(epochs=2000, plot_interval=100):
    X = torch.linspace(-2*np.pi, 2*np.pi, 200).unsqueeze(1)
    Y = torch.sin(X)

    model = AEONRegressor()
    loss_fn = nn.MSELoss()

    plt.ion()
    fig, ax = plt.subplots()

    prev_loss = None
    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, Y)

        # global signal = loss difference
        dL_signal = 0.0 if prev_loss is None else (loss.item() - prev_loss)
        model.evolve(loss, dL_signal)
        prev_loss = loss.item()

        if epoch % plot_interval == 0:
            ax.clear()
            ax.plot(X.detach().numpy(), Y.detach().numpy(), label="True Sine")
            ax.plot(X.detach().numpy(), pred.detach().numpy(), label=f"AEON Fit (epoch {epoch})")
            ax.set_ylim(-1.5, 1.5)
            ax.legend()
            plt.pause(0.1)
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    plt.ioff()
    plt.show()
    return model

if __name__ == "__main__":
    trained_model = train_aeon_sine()
```
