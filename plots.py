import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskMultiFidelityGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# Step 1: Generate sample training data
train_X = torch.rand(20, 3)  # 20 samples with (x1, x2, s)
train_Y = torch.sin((train_X[:, 0] + train_X[:, 1]) * torch.pi).unsqueeze(-1)  # target values

# Step 2: Initialize and train the SingleTaskMultiFidelityGP model
model = SingleTaskMultiFidelityGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)

# Step 3: Create a grid over x1 and x2 with a fixed fidelity level s
x1 = torch.linspace(0, 1, 50)
x2 = torch.linspace(0, 1, 50)
X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
s_fixed = torch.tensor([[1.0]])  # High-fidelity level

# Flatten the grid and concatenate with s_fixed
X_plot = torch.cat([
    X1.reshape(-1, 1),
    X2.reshape(-1, 1),
    s_fixed.expand(X1.numel(), 1)
], dim=1)

# Step 4: Evaluate the posterior mean and standard deviation with the trained model
with torch.no_grad():
    posterior = model.posterior(X_plot)
    mean = posterior.mean.reshape(50, 50).numpy()
    std = posterior.variance.sqrt().reshape(50, 50).numpy()

# Step 5: Plot the posterior mean and standard deviation
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot posterior mean
contour_mean = axs[0].contourf(X1.numpy(), X2.numpy(), mean, cmap='viridis')
axs[0].set_title("Posterior Mean")
fig.colorbar(contour_mean, ax=axs[0])

# Plot posterior standard deviation
contour_std = axs[1].contourf(X1.numpy(), X2.numpy(), std, cmap='viridis')
axs[1].set_title("Posterior Standard Deviation")
fig.colorbar(contour_std, ax=axs[1])

# Axis labels
for ax in axs:
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

plt.tight_layout()
plt.show()
