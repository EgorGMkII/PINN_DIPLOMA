# Sampling specification

## Point populations

Three populations are independent even when they reuse the same batch times:

- observations carry DNS velocity labels `(u,v,w)`;
- domain collocation points carry no labels and train PDE residuals;
- boundary points carry only the active RBC boundary targets.

T and p from DNS are hidden evaluation truth and must never enter an observation
batch in `EXP-001`.

## EXP-001 parity policy

To reproduce the article:

1. Globally shuffle DNS rows and take 4096 observation rows.
2. Copy their time coordinates to 4096 domain points.
3. Draw domain `x,y,z` independently from `Uniform(0,1)` every step.
4. Reuse batch times for independently sampled boundary points.
5. Generate one fresh sample per optimizer update.

Correctness fixtures contain explicit point tensors and perform no random draw.

## Stratified-time policy

This is the default post-parity observation policy. A batch allocates nearly
equal counts to each of the 11 times, samples rows without replacement within
each time stratum, and rotates any remainder across times. It controls temporal
coverage but does not replace the parity policy.

## Future residual-adaptive policy

Residual-adaptive sampling applies only to unlabeled domain collocation points.
It is out of scope for `EXP-001`. A future implementation must retain a fixed
uniform exploration fraction and must not use DNS T,p values to score points.
