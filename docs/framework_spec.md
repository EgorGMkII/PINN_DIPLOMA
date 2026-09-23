# Спецификация экспериментального PyTorch-фреймворка

Статус: архитектурный контракт до начала реализации.

## 1. Назначение

Фреймворк должен позволять менять по одной оси:

- PDE formulation (`VP`, `VV`, `FO`);
- observation operator;
- sampling policy и point budget;
- loss aggregation/weighting;
- optimizer и scheduler;
- eager/compiled execution;
- evaluation protocol.

Модульность не должна возвращать condition-level Python overhead в горячий шаг.
Конфигурация собирает модули до обучения; train step выполняет статически известный
набор активных операций.

## 2. Предлагаемая структура пакета

```text
src/diploma_pinn/
  models/             # field networks and initialization
  formulations/       # VP, VV, FO residual definitions
  operators/          # derivative primitives and field decoding
  observations/       # velocity data losses
  boundaries/         # RBC boundary constraints
  sampling/           # domain, boundary and time sampling
  losses/             # named reduction and weighting policies
  training/           # step builder, trainer, optimizer/scheduler factories
  evaluation/         # field, physics and performance metrics
  data/               # schemas, normalization and split loaders
  instrumentation/    # timers, memory and run manifests
configs/
tests/
scripts/
```

## 3. Главные интерфейсы

### Field model

```python
model(points: Tensor) -> Tensor
```

`points` имеет канонический порядок `(t,x,y,z)`. Имена и порядок выходов задаются
формулировкой и записываются в run manifest.

### PDE formulation

```python
formulation.residuals(model, points) -> dict[str, Tensor]
```

Formulation:

- декодирует outputs;
- запрашивает только необходимые производные;
- возвращает несвернутые именованные residuals;
- не генерирует точки;
- не применяет loss weights;
- не вызывает backward/optimizer;
- не пишет логи и файлы.

### Sampler

```python
sampler.sample(step_context) -> PointBatch
```

`PointBatch` раздельно содержит data, domain и активные boundary points. Все
тензоры создаются сразу на целевом device. Point counts являются явной частью
конфига и manifest, а не выводятся неявно из числа conditions.

### Observation operator

```python
observation.loss(model, observations) -> dict[str, Tensor]
```

Первый обязательный оператор -- MSE наблюдаемых `(u,v,w)` в лагранжевых точках.
T и p не читаются как train labels в velocity-only экспериментах.

### Loss assembler

```python
loss_assembler(residuals, observation_losses, boundary_losses) -> LossReport
```

Он единолично определяет reduction и weights. `LossReport.total` -- scalar для
backward; detached logging values создаются только после вычислительного шага.

### Step engine

```python
step_engine.step(data_batch, step_index) -> StepMetrics
```

Внутри выполняются sampling, active forward/residuals, один total loss, один
backward и один optimizer update. В production path запрещены циклы по абстрактным
conditions, динамическое обнаружение residuals и CPU synchronization.

## 4. Модульность без потери скорости

Модули выбираются до построения hot step:

```python
step = build_train_step(
    model=model,
    formulation=VP(...),
    sampler=sampler,
    observation=velocity_observation,
    boundaries=active_boundaries,
    loss_assembler=loss_assembler,
    optimizer=optimizer,
)
```

`build_train_step` создает специализированную функцию для данной конфигурации.
Нулевые loss weights удаляют ветвь при сборке, а не умножают уже вычисленный
residual на ноль. Это сохраняет расширяемость вне шага и монолитность внутри него.

## 5. Семантика шага, наследуемая от article_pinn

```text
data minibatch
  -> sample current domain/boundary points on GPU
  -> data forward
  -> formulation forward and required coordinate derivatives
  -> active boundary forward/derivatives
  -> named reductions and one total scalar
  -> one backward
  -> one optimizer update
  -> detached instrumentation outside the hot path
```

Для parity-профиля фиксируются Adam, reductions, batch size, sine architecture,
initialization, weights и LR policy reference. После parity отдельными флагами
исследуются fused Adam, compilation и alternative derivative backends.

## 6. Оптимизация как отдельный модуль

`OptimizerSpec` задает class и все параметры явно. Никакие framework defaults не
должны незаметно различаться. Для TensorFlow parity, как минимум:

```text
Adam(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-7, weight_decay=0)
ReduceLROnPlateau(factor=0.8, patience=100, min_lr=1e-4, min_delta=5e-6)
```

Optimizer, scheduler, gradient accumulation, clipping, AMP и compilation --
независимые конфигурационные поля. Основное VP--VV--FO сравнение не меняет их.

## 7. Профили исполнения

- `parity`: максимально близко к TensorFlow reference, без оптимизаций PyTorch.
- `eager`: специализированный PyTorch step, eager autograd.
- `compiled`: компиляция целого поддерживаемого loss/step, с проверкой gradients.
- `profile`: синхронизированные таймеры по фазам и memory statistics.

Компиляция считается допустимой только после теста совпадения outputs, residuals,
parameter gradients и update. Компиляция одной модели не считается компиляцией
PINN step.

## 8. Обязательные тесты

- shape/device/dtype contracts каждого модуля;
- analytical residuals для VP, VV, FO;
- finite-difference spot checks выбранных coordinate derivatives;
- TensorFlow--PyTorch parity на фиксированных весах и точках;
- loss reduction и zero-weight pruning;
- deterministic sampling при seed;
- eager/compiled gradient parity;
- checkpoint round-trip и run-manifest completeness.

## 9. Инварианты воспроизводимости

Каждый run фиксирует:

```text
git commit and dirty flag
resolved config
dataset fingerprint and split id
seed and deterministic settings
device, GPU, precision and library versions
model parameter count and output schema
points per type per optimizer step
number of updates
compile and warm-up policy
wall time and peak memory
```

Термин `epoch` может использоваться для UI, но не является основной единицей
сравнения между реализациями.
# Current formulation and analysis modules

The formulation switch is `physics.formulation: vp|vv|fo`. VP, VV and FO own
their derivative operators and loss kernels; switching one does not modify the
others. FO uses 17 outputs with `(u,v,w,T,p)` first so common evaluation remains
compatible.

VV pressure recovery is post-processing: the trained VV model is frozen and a
scalar sine MLP fits the momentum-implied pressure gradient. Its state is stored
inside the single final experiment checkpoint.

`analysis.observability` evaluates prescribed temperature modes without a
training run. `VelocityDatasetView` owns deterministic DNS masks and synthetic
velocity noise. PTV CSV loading exposes only coordinates and velocities.

Every training run writes local JSONL metrics. Optional W&B, evaluation and
profiling stay outside the differentiable loss kernel.

