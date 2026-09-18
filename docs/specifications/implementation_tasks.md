# EXP-001 implementation task board

Each task should be completed with its focused tests. Do not launch full training
from this board.

- [x] Create package skeleton and shared contracts.
- [x] Add reference sine MLP and loss-weight skeleton.
- [x] Add guarded DNS loader.
- [x] Create interface skeletons for every EXP-001 module.
- [x] Specify autograd, training, evaluation, runtime, parity, and benchmark contracts.
- [x] Close the YAML-to-report skeleton with config, data validation, parity, benchmark, and device modules.
- [x] Implement deterministic global-shuffle observation sampler.
- [x] Implement stratified-time observation sampler.
- [x] Implement uniform domain and active RBC boundary samplers.
- [x] Implement derivative primitives and VP residual kernel.
- [x] Implement velocity observation and active boundary losses.
- [x] Build the concrete EXP-001 loss kernel with reference reductions.
- [x] Implement explicit Adam, plateau scheduler, device/dtype, and seed policy.
- [x] Assemble the local one-step VP vertical slice from typed config.
- [ ] Add TensorFlow weight import/export fixture tooling.
- [ ] Add output, derivative, residual, gradient and Adam-update parity tests.
- [x] Add smoke-ready resolved configuration and run manifest schemas.
- [x] Add evaluator metrics with pressure gauge alignment.
- [ ] Add W&B adapter outside the hot step.
- [x] Add dataset validation, smoke CLI, and first DataSphere GPU job config.
- [ ] Add eager benchmark CLI and full DataSphere training/benchmark job configs.
- [ ] Add compiled/fused profiles only after correctness gates pass.
