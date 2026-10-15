"""A multilayer perceptron on raw pixels.

TASK 2. Build `MLP`: a fully connected network that maps one flattened,
standardised photograph (`in_dim` grey levels) to one number in [0, 1].

* `hidden` is a list of layer widths from the config, for example [256, 64];
  build one Linear + nonlinearity + Dropout block per entry.
* `dropout` is the probability from the config. Where it goes, and whether
  it belongs after the last hidden layer, is your call.
* The output must lie in [0, 1]. `prior` is the mean target of the training
  rows; it is passed to you for a reason you have to find yourself.
* `forward` returns shape (batch,), not (batch, 1).

It trains in seconds on a CPU. That is the point of starting here: every
decision in it can be tested in the time it takes to read this docstring.
Count its parameters against the number of training photographs before you
decide how much dropout it needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.3, prior: float = 0.5):
        super().__init__()

        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = width

        self.trunk = nn.Sequential(*layers)

        # Output head: one logit mapped to [0, 1] via sigmoid.
        # Initialise the bias so sigmoid(bias) ≈ prior, which means the
        # network's first predictions are already near the training mean
        # rather than 0.5, shrinking the initial loss and stabilising early
        # gradient steps.
        self.head = nn.Linear(prev, 1)
        nn.init.zeros_(self.head.weight)
        prior_clamped = float(torch.tensor(prior).clamp(1e-6, 1 - 1e-6))
        self.head.bias.data.fill_(torch.log(torch.tensor(prior_clamped / (1 - prior_clamped))).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.trunk(x))).squeeze(1)  # (batch,)
