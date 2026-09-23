# EXP-002: matched VP--VV pilot

Status: implementation ready; remote runs pending.

## Question

Does removing pressure from the training graph improve hidden-temperature recovery enough to justify direct VV third derivatives?

## Controlled comparison

VP and VV use the same DNS fingerprint, velocity-only observations, time-stratified batches of 4096, 10x256 sine MLP, Adam settings, scheduler, seed and optimizer-step count. VP outputs `(u,v,w,T,p)`; VV outputs `(u,v,w,T)` and computes `omega=curl(u)`.

DNS `T,p` are evaluator-only. VV pressure is reconstructed after training by a separate scalar network fitted to the momentum-implied pressure gradient, never to DNS pressure.

## Runs and gate

1. Matched VP and VV benchmarks: 20 warmup and 50 measured steps each.
2. Matched VP/VV runs: 7,040 updates (10 full DNS passes).
3. VV pressure-recovery pilot.
4. Only after passing the gate: 140,800-update runs, first seed 2204 and then 2205/2206.

Direct VV is recorded as impractical if it is non-finite, does not fit batch 4096, or its warm-step exceeds five times VP. Do not hide this result by reducing only the VV point budget.

## Outputs

Compare hidden-field curves every 10 epochs, final field/mode diagnostics, independent PDE residuals, warm-step phase times and peak CUDA memory. Save one final checkpoint only.
