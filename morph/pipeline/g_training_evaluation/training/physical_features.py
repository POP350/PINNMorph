"""Physical coordinate features shared by the base network and online branches."""
import math

import torch
from torch import nn


class SpatialPeriodicTransform(nn.Module):
    """Replace periodic coordinates by sin/cos; retain nonperiodic coordinates."""

    def __init__(self, input_dim, axes, periods, origins=None):
        super().__init__()
        if len(axes) != len(periods) or not axes or len(set(axes)) != len(axes):
            raise ValueError("Periodic axes and periods must be nonempty and one-to-one")
        if any(type(a) is not int or not 0 <= a < input_dim for a in axes):
            raise ValueError("Periodic axis outside physical input dimensions")
        if any(not math.isfinite(p) or p <= 0 for p in periods):
            raise ValueError("Periodic lengths must be finite and positive")
        origins = [0.0] * len(axes) if origins is None else origins
        if len(origins) != len(axes) or any(not math.isfinite(v) for v in origins):
            raise ValueError("Invalid periodic origins")
        self.axes = tuple(axes)
        self.raw_axes = tuple(a for a in range(input_dim) if a not in axes)
        self.output_dim = input_dim + len(axes)
        self.register_buffer("periods", torch.tensor(periods))
        self.register_buffer("origins", torch.tensor(origins))

    def forward(self, inputs):
        phase = 2 * math.pi * (inputs[:, self.axes] - self.origins) / self.periods
        return torch.cat((inputs[:, self.raw_axes], phase.sin(), phase.cos()), dim=-1)


def directional_frequency_bank(input_dim, count, axis_frequencies):
    """Cycle over supplied physical frequencies (cycles / coordinate unit)."""
    entries = []
    for axis, values in sorted(axis_frequencies.items(), key=lambda item: int(item[0])):
        axis = int(axis)
        if not 0 <= axis < input_dim or not values:
            raise ValueError("Invalid Fourier axis")
        for value in values:
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Fourier frequencies must be finite and positive")
            entries.append((axis, float(value)))
    if not entries:
        raise ValueError("Directional frequencies cannot be empty")
    result = torch.zeros(count, input_dim)
    for i in range(count):
        axis, value = entries[i % len(entries)]
        result[i, axis] = value
    return result
