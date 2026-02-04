import torch
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures/surfaces", exist_ok=True)
os.makedirs("figures/gradients", exist_ok=True)

torch.manual_seed(42)

W1 = torch.randn(50, 2, dtype=torch.float32)
b1 = torch.randn(50, dtype=torch.float32)
W2 = torch.randn(1, 50, dtype=torch.float32)
b2 = torch.randn(1, dtype=torch.float32)

xs = np.linspace(-2, 2, 200)
ys = np.linspace(-2, 2, 200)


def activation(z, kind):
    z = z.float()
    if kind == "linear":
        return z
    if kind == "relu":
        return torch.relu(z)
    if kind == "swish":
        return z * torch.sigmoid(z)
    if kind == "mish":
        return z * torch.tanh(torch.log1p(torch.exp(z)))
    raise ValueError("Unknown activation")


def surface(kind):
    Z = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            inp = torch.tensor([x, y], dtype=torch.float32)
            h = activation(W1 @ inp + b1, kind)
            out = W2 @ h + b2
            Z[i, j] = out.item()
    return Z


def gradient_norm(kind):
    Z = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            inp = torch.tensor([x, y], dtype=torch.float32, requires_grad=True)
            h = activation(W1 @ inp + b1, kind)
            out = W2 @ h + b2
            out.backward()
            Z[i, j] = inp.grad.norm().item()
    return Z


activations = ["linear", "relu", "swish", "mish"]

for act in activations:
    # surface
    Z = surface(act)
    plt.imshow(Z, extent=[-2,2,-2,2], origin="lower", cmap="viridis")
    plt.colorbar()
    plt.title(f"{act.capitalize()} Output Surface")
    plt.savefig(f"figures/surfaces/{act}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # gradient
    Zg = gradient_norm(act)
    plt.imshow(Zg, extent=[-2,2,-2,2], origin="lower", cmap="inferno")
    plt.colorbar()
    plt.title(f"{act.capitalize()} Gradient Norm")
    plt.savefig(f"figures/gradients/{act}_grad.png", dpi=300, bbox_inches="tight")
    plt.close()
