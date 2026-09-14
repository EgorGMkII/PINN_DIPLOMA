# Training lifecycle specification

## Construction

Configuration is parsed and validated once. The factory resolves device, dtype,
model, sampling policy, active loss branches, optimizer, scheduler, evaluator,
tracker, checkpoint policy, and profiler before the first step. Unknown keys and
unsupported combinations are errors.

## Step

Each optimizer step obtains one observation batch, samples matching domain and
active boundary points, zeroes gradients, evaluates one specialized total-loss
kernel, calls one backward, calls one optimizer update, then detaches a small
logging payload.

No evaluation, checkpoint, W&B operation, JSON write, `.cpu()`, or device
synchronization occurs inside the normal hot step. Profile mode may synchronize
explicitly and must label those measurements.

The trainer owns step limits, scheduler timing, sparse evaluation, sparse
checkpointing, signal-aware shutdown, tracker finalization, and final summary.
Runs compare optimizer updates and wall time. Epoch is display metadata only.

Checkpoints contain schema version, model, optimizer, scheduler where active,
step, RNG states, and config fingerprint. Resume rejects incompatible schemas.
