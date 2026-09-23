# EXP-003: temperature-mode observability

Status: implemented; analysis run pending.

## Question

Which temperature modes are visible to VP momentum and VV vorticity residuals, and which can be absorbed by pressure?

## Procedure

Use `diploma-pinn observability` on a fixed, time-stratified DNS coordinate subset. Inject unit modes that vanish at the thermal walls:

- `sin(n*pi*z)`, n=1..3;
- `sin(pi*z) cos(2*pi*k*x)`, k=1..3;
- `sin(pi*z) cos(2*pi*k*y)`, k=1..3.

For each mode report RMS of the optimally vertically integrated VP pressure-compensated momentum perturbation, VV buoyancy curl, energy diffusion residual and boundary violation. No network is trained and no DNS `T,p` values are read.

## Interpretation

A plane-constant vertical mode has zero buoyancy curl and can be exactly represented by a hydrostatic pressure correction, while horizontal modes leave VP and VV signatures. Training diagnostics project reconstructed temperature fluctuations onto the same first three horizontal modes.

## First numerical check (2026-09-23)

The 65,536-point fixed DNS subset confirmed the analytical structure. For
vertical modes 1--3, pressure-compensated VP momentum RMS was about `1e-8` and
VV buoyancy-curl RMS was exactly zero. For horizontal modes 1--3, VP RMS grew
approximately `0.576, 1.154, 1.728`, while VV RMS grew approximately
`1.785, 3.58, 5.35`. Energy residual increased with wave number and all thermal
wall violations stayed at floating-point zero.
