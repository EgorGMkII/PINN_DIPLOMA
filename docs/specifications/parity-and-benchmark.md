# Parity and benchmark specification

The parity fixture stores small explicit point groups and framework-neutral Dense
weights. Check order: parameters, outputs, first derivatives, second derivatives,
residuals, reductions, total loss, parameter gradients, then one Adam update.
Reports include max absolute error, relative L2, tolerances, and failing tensor
names. Tolerances may not be widened silently.

The benchmark has separate warm-up/compile and measured phases. CUDA timings
synchronize at interval boundaries. It reports sampling, data forward, physics
derivatives, boundary, assembly, backward, optimizer, total step, and peak memory.
Microbenchmarks disable W&B, checkpointing, plots, and full-field evaluation.
