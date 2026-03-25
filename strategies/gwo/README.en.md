# Enhanced Grey Wolf Optimizer for GP

**中文说明：** [README.md](README.md)

## Summary

An enhanced Grey Wolf Optimizer (GWO) tailored for Genetic Programming. It combines phased exploration–exploitation schedules, adaptive layer sizing, and distance-aware selection to improve runtime and solution quality.

## Theory

For detailed math, see [FORMULAS.md](./FORMULAS.md) (if present in your checkout).

### 1. Phased exploration vs. exploitation

The run is split into three phases (tuned by default for ~50 generations):

- **Exploration (≈0–20% of gens):** Large convergence factor (a ≈ 2.0) favors global search.  
- **Balanced (≈20–70%):** Linear decay of a from ~2.0 toward ~0.1.  
- **Exploitation (≈70–100%):** Small a (~0.1) emphasizes local refinement.

### 2. Layered leadership

Inspired by the wolf hierarchy:

- **Alpha (~5%)** — best individual, primary leader  
- **Beta (~10%)** — second best  
- **Delta (~15%)** — third tier  
- **Omega (~70%)** — remaining individuals updated from leaders  

### 3. Adaptive layer ratios

Population diversity drives ratio updates:

- **Low diversity** — enlarge the Omega share to boost exploration.  
- **High diversity** — emphasize leader tiers for exploitation.

### 4. Smart position updates

Selection pressure depends on the phase:

- **Exploration** — prefer moderately distant candidates to preserve diversity.  
- **Balanced** — exponential decay biased toward leaders.  
- **Exploitation** — stronger decay, tight clustering around leaders.

## Performance notes

1. Optional fitness caching to avoid duplicate evaluations.  
2. Dynamic elitism tuned by phase.  
3. Selection roughly **O(n log n)**, suitable for large populations.

## Enabling GWO in config

In your GP config (e.g. `main.py`):

```python
"selection": {
    "strategy": "tournament",  # ignored when enable_gwo=True
    "tournament_size": 7,
    "hof_size": 20,
    "enable_gwo": True,
    "gwo_type": "enhanced",  # "traditional" or "enhanced"
    "gwo_config": {
        "tau1": 0.2,
        "tau2": 0.7,
        "a0": 2.0,
        "am": 1.0,
        "af": 0.1,
        "rho_alpha": 0.05,
        "rho_beta": 0.10,
        "rho_delta": 0.15,
        "use_adaptive_layers": True,
        "diversity_threshold": 0.15,
        "adaptive_factor": 0.1
    }
}
```

### Defaults

If `gwo_config` is omitted, built-in defaults target ~50 generations and large populations (e.g. 1024); adjust for your experiment.

## Advantages

1. Faster convergence via phase-aware balance.  
2. Better final fitness through adaptive leadership.  
3. Lower overhead from caching + efficient selection.  
4. Grounded in classical GWO with GP-specific hooks.

## Parameter reference

### Phase thresholds
- `tau1` (default 0.2) — end of exploration phase (fraction of progress).  
- `tau2` (default 0.7) — end of balanced phase.

### Convergence factors
- `a0` (default 2.0) — initial a (exploration strength).  
- `am` (default 1.0) — mid-run a.  
- `af` (default 0.1) — final a (exploitation strength).

### Layer ratios
- `rho_alpha`, `rho_beta`, `rho_delta` — Alpha/Beta/Delta fractions.  
- Omega share = `1 - (rho_alpha + rho_beta + rho_delta)`.

### Adaptivity
- `use_adaptive_layers` — toggle adaptive ratios.  
- `diversity_threshold` — diversity trigger.  
- `adaptive_factor` — step size for ratio updates.

## Experiment tips

1. Start from defaults, then tweak `tau1` / `tau2` and `{a0, am, af}`.  
2. Compare against tournament selection on the same seeds.  
3. Log diversity metrics to validate adaptive behavior.

## References

- Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). *Grey Wolf Optimizer.* Advances in Engineering Software, 69, 46–61.  
- Related work on staged dynamic policies in evolutionary algorithms.
