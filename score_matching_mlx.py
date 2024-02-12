import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from sklearn.datasets import make_swiss_roll

from helper_plot import hdr_plot_style

hdr_plot_style()


# Sample a batch from the swiss roll
def sample_batch(size, noise=0.5):
    x, _ = make_swiss_roll(size, noise=noise)
    return x[:, [0, 2]] / 10.0


# Plot it
data = sample_batch(10**4).T
plt.figure(figsize=(16, 12))
plt.scatter(*data, alpha=0.5, color="red", edgecolor="white", s=40)


def sliced_score_matching(model, samples):
    samples.requires_grad_(True)
    # Construct random vectors
    vectors = mlx.randn_like(samples)
    vectors = vectors / mlx.norm(vectors, dim=-1, keepdim=True)
    # Compute the optimized vector-product jacobian
    logp, jvp = mx.jvp(model, samples, vectors)
    # Compute the norm loss
    norm_loss = (logp * vectors) ** 2 / 2.0
    # Compute the Jacobian loss
    v_jvp = jvp * vectors
    jacob_loss = v_jvp
    loss = jacob_loss + norm_loss
    return loss.mean(-1).mean(-1)


def denoising_score_matching(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + mx.random.normal(samples.shape) * sigma
    target = -1 / (sigma**2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.reshape(target.shape[0], -1)
    scores = scores.reshape(scores.shape[0], -1)
    loss = 1 / 2.0 * ((scores - target) ** 2).sum(axis=-1).mean(axis=0)
    return loss


# Our approximation model
model = nn.Sequential(
    nn.Linear(2, 128),
    nn.Softplus(),
    nn.Linear(128, 128),
    nn.Softplus(),
    nn.Linear(128, 128),
    nn.Softplus(),
    nn.Linear(128, 2),
)

# Create ADAM optimizer over our model
optimizer = optim.Adam(learning_rate=1e-3)
dataset = mx.array(data.T)

loss_and_grad_fn = nn.value_and_grad(model, denoising_score_matching)

for t in range(5000):
    # Compute the loss.
    loss, grads = loss_and_grad_fn(model, dataset)
    # Calling the step function to update the parameters
    optimizer.update(model, grads)
    # Print loss
    if (t % 1000) == 0:
        print(loss)


def plot_gradients(model, data, plot_scatter=True):
    xx = np.stack(
        np.meshgrid(np.linspace(-1.5, 2.0, 50), np.linspace(-1.5, 2.0, 50)), axis=-1
    ).reshape(-1, 2)
    scores = np.array(model(mx.array(xx)), copy=False)
    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
    # Perform the plots
    plt.figure(figsize=(16, 12))
    if plot_scatter:
        plt.scatter(*data, alpha=0.3, color="red", edgecolor="white", s=40)
    plt.quiver(
        xx.T[0],
        xx.T[1],
        scores_log1p[:, 0],
        scores_log1p[:, 1],
        width=0.002,
        color="white",
    )
    plt.xlim(-1.5, 2.0)
    plt.ylim(-1.5, 2.0)
    plt.show()


plot_gradients(model, data)
