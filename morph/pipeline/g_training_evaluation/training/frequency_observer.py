"""Reference-free directional spectra on fixed spatial residual slices."""
import torch


def residual_slice_spectra(loss, model, count=64):
    spec = loss.problem.get_spec()
    variables = list(spec.input_variables)
    bounds = [spec.domain.bounds[v] for v in variables]
    device = next(model.parameters()).device
    result, total_points = {}, 0
    for axis, name in enumerate(variables):
        if name == "t":
            continue
        lines = []
        for fraction in (0.25, 0.5, 0.75):
            points = torch.tensor([[lo + fraction * (hi - lo) for lo, hi in bounds]], device=device).repeat(count, 1)
            lo, hi = bounds[axis]
            points[:, axis] = lo + (torch.arange(count, device=device) + 0.5) * (hi - lo) / count
            residuals = loss.evaluate_registered_pde_residuals(model, points)
            for residual in residuals.values():
                values = residual.reshape(count, -1)
                spectrum = torch.fft.rfft(values - values.mean(dim=0), dim=0).abs().square().sum(dim=1)
                lines.append(spectrum)
            total_points += count
        if not lines:
            continue
        energy = torch.stack(lines).mean(dim=0)
        energy[0] = 0
        total = float(energy.sum())
        result[name] = {"axis": axis, "peak_cycles_per_unit": float(energy.argmax()) / (bounds[axis][1] - bounds[axis][0]) if total > 1e-20 else None,
            "high_band_energy_fraction": float(energy[3:].sum()) / max(total, 1e-20),
            "nonconstant_energy": total, "samples_per_slice": count, "slice_count": 3,
            "source": "fixed_governing_residual_slices", "reference_solution_used": False}
    return result, total_points
