# EXP-004: mixed first-order pilot

Status: implementation ready; remote run pending.

## Formulation

The output order is `(u,v,w,T,p,G_ux,G_uy,G_uz,G_vx,G_vy,G_vz,G_wx,G_wy,G_wz,q_x,q_y,q_z)`.

Momentum and energy use `G` and `q`; compatibility residuals enforce `G=grad(u)` and `q=grad(T)`. Continuity is `tr(G)=0`. Only first coordinate derivatives of network outputs are used.

Compatibility losses use weights 0.1 for `G` and 0.01 for `q`, matching the momentum and energy group scales. These weights are frozen for the first pilot.

## Gate

Run analytical residual tests, then the same 10-epoch DNS pilot and benchmark contract as EXP-002. A 200-epoch run is allowed only if FO is finite, fits the matched batch and differs meaningfully in memory, speed or hidden-field convergence.
