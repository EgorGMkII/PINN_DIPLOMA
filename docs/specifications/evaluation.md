# Evaluation specification

Evaluation runs outside training autograd and predicts the complete DNS dataset
in bounded chunks. It computes MSE and relative L2 independently for u,v,w,T,p.
The evaluator may read DNS T,p; training modules may not.

Pressure is identifiable only up to an additive gauge. Before pressure metrics,
subtract the predicted-minus-reference mean independently for each unique time.
Report both the gauge convention and per-time aggregate. Do not train against DNS
pressure to resolve the gauge in velocity-only experiments.

Validation splits must be defined by whole times, spatial regions, or explicit
subsampling masks. A random neighbouring-row split is not sufficient evidence of
generalization for the regular 64^3 DNS sequence.
