# EXP-005: sparse and noisy velocity observations

Status: infrastructure ready; method selection pending EXP-002/004.

## Design

Run only VP and the formulation selected by the pilot. Use seeds 2204, 2205 and 2206 with identical masks between methods.

Observation levels are defined by counts, not filename percentages:

- dense DNS: 262,144 velocities per time;
- `005`-matched: 40,500 per time;
- `001`-matched: 8,100 per time.

The same deterministic time-stratified mask is reused for paired methods. Noise is additive in normalized velocity units; its standard deviation must first be estimated and recorded from the PTV preprocessing diagnostics, then frozen in the run config. Do not tune it from reconstruction quality.

## Metrics

Report field errors, temperature profile/fluctuations/Fourier modes, pressure hydrostatic/dynamic components, heat transport, independent PDE residuals and across-seed dispersion. DNS `T,p` remain evaluation-only.
