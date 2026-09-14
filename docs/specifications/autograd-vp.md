# VP autograd and loss specification

## Graph boundary

One call to the concrete VP loss kernel must produce one connected scalar graph.
The data, domain, and active boundary forwards may use different point tensors,
but all active losses are summed before the single `backward()` call. A residual
module must never call `backward`, detach tensors, transfer to CPU, or synchronize.

`compute_vp_derivatives` owns coordinate differentiation. Input domain points
require gradients before the model forward. It returns fields, all required first
derivatives, and only diagonal spatial second derivatives required by VP. Mixed
derivatives and `p_t` are not requested.

## Required derivatives

```text
u,v,w,T: t,x,y,z
p:       x,y,z
u,v,w,T: xx,yy,zz
```

Derivative keys use `<field>_<coordinate>` and `<field>_<coordinate><coordinate>`.
The output field order remains `(u,v,w,T,p)`.

## Reductions

- `data`: mean of squared errors jointly over u,v,w and rows.
- `momentum`: mean of squares jointly over r_u,r_v,r_w and rows.
- `energy`: mean square of r_T.
- `continuity`: mean square of r_div.
- `boundary`: literal weighted sum of reference boundary component reductions.

Tests must cover an analytical manufactured field, finite-difference spot checks,
TensorFlow fixture parity, parameter gradients, absence of unused derivatives,
and finite outputs for an ordinary float32 batch.
