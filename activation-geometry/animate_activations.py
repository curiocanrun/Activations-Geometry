import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

os.makedirs("figures/animations", exist_ok=True)

torch.manual_seed(42)

W1 = torch.randn(50, 2, dtype=torch.float32)
b1 = torch.randn(50, dtype=torch.float32)
W2 = torch.randn(1, 50, dtype=torch.float32)
b2 = torch.randn(1, dtype=torch.float32)

xs = np.linspace(-2, 2, 150)
ys = np.linspace(-2, 2, 150)
frames = 31

X, Y = torch.meshgrid(torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32), indexing="ij")
XY = torch.stack([X, Y], dim=-1)   
XY_flat = XY.reshape(-1, 2) 

def blended_activation(z, alpha):
    z = z.float()
    relu = torch.relu(z)
    mish = z * torch.tanh(torch.log1p(torch.exp(z)))
    return (1 - alpha) * relu + alpha * mish

def surface(alpha):
    with torch.no_grad():
        H = blended_activation(XY_flat @ W1.T + b1, alpha)  
        out = H @ W2.T + b2                                 
    Z = out.reshape(len(xs), len(ys)).numpy()
    return Z

fig, ax = plt.subplots()
im = ax.imshow(surface(0), extent=[-2,2,-2,2], origin="lower", cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()

def update(frame):
    alpha = frame / 50
    im.set_data(surface(alpha))
    ax.set_title(f"ReLU -> Mish (alpha = {alpha:.2f})")
    return [im]

ani = FuncAnimation(fig, update, frames=51, blit=True)
writer = PillowWriter(fps=15)
ani.save("figures/animations/relu_to_mish.gif", writer=writer)
plt.close()
