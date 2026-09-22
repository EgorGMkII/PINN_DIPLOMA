# Evaluation specification

Evaluation runs outside training autograd and predicts the complete DNS dataset
in bounded chunks. It computes MSE and relative L2 independently for u,v,w,T,p.
The evaluator may read DNS T,p; training modules may not.

Pressure is identifiable only up to an additive gauge. Before pressure metrics,
subtract the predicted-minus-reference mean independently for each unique time.
Report both the gauge convention and per-time aggregate. Do not train against DNS
pressure to resolve the gauge in velocity-only experiments.

## Final-run artifacts

Every non-smoke training run saves `checkpoints/final.pt` after the final
optimizer update. The checkpoint includes the model, optimizer, optional
scheduler, RNG states, the completed step, and a resolved-config fingerprint.

When full DNS evaluation is enabled, `diagnostics.json` additionally records
horizontal-plane diagnostics indexed by `(t,z)`:

- mean temperature profile and its reconstruction metrics;
- temperature fluctuations about that profile;
- gauge-aligned plane-mean (hydrostatic) pressure and dynamic pressure;
- plane-mean convective heat flux `w*T`.

These diagnostics are evaluation-only: DNS `T,p` never enter the training loss.
For direct VV, the network has no pressure output, so both pressure diagnostics
are `null`; pressure must not be silently inferred or scored in that run.
For incomplete PTV-like coverage, absent `(t,z)` planes are marked in the
`valid` mask and excluded from profile metrics.

Validation splits must be defined by whole times, spatial regions, or explicit
subsampling masks. A random neighbouring-row split is not sufficient evidence of
generalization for the regular 64^3 DNS sequence.
