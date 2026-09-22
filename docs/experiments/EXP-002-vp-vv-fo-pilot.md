# EXP-002: VP--VV--FO formulation pilot

Status: `proposed`

Created: 2026-09-22
Related global stage: E2

## Question

How does the mathematical representation of the Boussinesq equations affect
the recovery of hidden temperature and pressure fields from the same
velocity-only DNS observations?

## Hypothesis

The velocity--vorticity (`VV`) residual removes pressure from the physics loss
and constrains horizontal temperature gradients through the curl of buoyancy.
It may improve temperature reconstruction, but its direct implementation needs
third coordinate derivatives of velocity. The mixed first-order (`FO`) residual
avoids such high-order coordinate derivatives at the price of compatibility
constraints for predicted gradients.

## Methods Compared

| Method | Network output | Physics residual |
| --- | --- | --- |
| VP | `(u,v,w,T,p)` | momentum, energy, continuity |
| VV | `(u,v,w,T)` | vorticity transport, energy, continuity |
| FO | `(u,v,w,p,T,G,q)` | first-order momentum/energy plus `G=grad(u)`, `q=grad(T)` |

The primary independent variable is the PDE formulation. The VP implementation
and its 200-epoch DNS result are the baseline. Architecture, observations,
sampling, optimizer, learning-rate policy, seeds, and point budgets remain
matched as far as output schemas allow.

## VV Equations

Let `omega = curl(u)`, `nu = sqrt(Pr/Ra)`, and
`kappa = sqrt(1/(Pr*Ra))`. With gravity in `+z`, the residuals are

```text
r_omega = omega_t + (u . grad) omega - (omega . grad) u
          - nu Laplace(omega) - curl(T e_z)
r_T     = T_t + (u . grad) T - kappa Laplace(T)
r_div   = div(u)

curl(T e_z) = (T_y, -T_x, 0).
```

The direct VV implementation deliberately computes `omega=curl(u)` by
autograd. Therefore `Laplace(omega)` requires third derivatives of `u`.
This is part of the method being tested. It must not be replaced by an
auxiliary-vorticity network without a separate experiment specification.

## Data Contract

Primary data is `RBC_PTV_1E6_07_t_11.npz`, fingerprint
`7da325eb40b5a1f86dff57a90adfd107eefb9a830852e6a8716c128d61e76615`.
Only `(u,v,w)` may enter the observation loss. DNS `T,p` are evaluation truth.
The direct VV model has no pressure output; pressure diagnostics therefore apply
only to VP and FO unless a separately specified pressure-recovery procedure is
implemented.

## Pilot Procedure

1. Analytical residual tests for each formulation.
2. One short fixed-seed DNS run per method to reject invalid or impractical
   variants.
3. Compare temperature profile, temperature fluctuations, convective heat flux,
   velocity error, independent PDE residuals, warm-step time, and peak memory.
4. Run 200 epochs only for methods that pass the pilot and save a final
   checkpoint plus `diagnostics.json`.

## Acceptance And Stop Criteria

- Every manufactured residual test must pass before training.
- A method that produces non-finite losses or cannot complete the fixed pilot on
  the same GPU is recorded as infeasible rather than retuned indefinitely.
- Pressure is not treated as a VV metric without an explicit recovery method.
- No sparse/noisy matrix is launched until VP, VV, and FO pilot results select
  the informative methods.

## Expected Artifacts

Each run writes a manifest, resolved config, summary, final checkpoint,
`diagnostics.json`, and a profiler summary. Large predicted fields remain
ignored artifacts.
