import torch
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures/loss", exist_ok=True)

torch.manual_seed(0)

X = torch.randn(200, 2, dtype=torch.float32)
y = torch.sin(X[:,0]) + torch.cos(X[:,1])
y = y.float()

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

def loss_surface(kind):
    w1 = np.linspace(-3, 3, 200)
    w2 = np.linspace(-3, 3, 200)
    Z = np.zeros((len(w1), len(w2)))

    for i, a in enumerate(w1):
        for j, b in enumerate(w2):
            W = torch.tensor([[a, b]], dtype=torch.float32)
            preds = activation(X @ W.T, kind).squeeze()
            Z[i, j] = torch.mean((preds - y) ** 2).item()

    return Z, w1, w2

for act in ["linear", "relu", "swish", "mish"]:
    Z, w1, w2 = loss_surface(act)
    plt.imshow(Z, extent=[-3,3,-3,3], origin="lower", cmap="plasma")
    plt.colorbar()
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.title(f"{act.capitalize()} Loss Surface")
    plt.savefig(f"figures/loss/{act}_loss.png", dpi=300, bbox_inches="tight")
    plt.close()
