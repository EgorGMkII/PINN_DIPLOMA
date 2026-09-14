# EXP-001: TensorFlow--PyTorch VP parity

Status: `proposed`

Created: 2026-09-14
Global stage: E0--E1
Primary formulation: strong velocity--pressure (`VP`)

## 1. Вопрос

Можно ли воспроизвести в новом специализированном PyTorch-ядре математическую и
вычислительную семантику TensorFlow `article_pinn/RBC-PINN`, а затем получить
сопоставимую или лучшую стоимость шага без изменения VP-задачи?

## 2. Гипотезы

### H1. Численная parity

При одинаковых весах сети, входных data/domain/boundary points, dtype и формулах
TensorFlow и PyTorch дают совпадающие outputs, необходимые производные, PDE
residuals и scalar losses в заранее заданном tolerance.

### H2. Parity optimizer step

После явного согласования варианта Adam и его состояния один optimizer update
PyTorch воспроизводит TensorFlow update в объяснимом tolerance. Если реализации
Adam алгоритмически различаются, это фиксируется отдельно от autograd parity.

### H3. Стоимость специализированного шага

После достижения correctness parity PyTorch-step, собранный заранее из активных
модулей, не содержит condition-level Python orchestration и позволяет локализовать
или устранить прежнее многократное отставание по времени шага.

## 3. Научная и инженерная ценность

EXP-001 сам по себе не является научной новизной диплома. Он создает достоверный
измерительный baseline для последующего VP--VV--FO исследования.

Если parity достигнута, но PyTorch существенно медленнее, причина находится в
backend/AD/исполнении. Если parity не достигается, прошлые сравнения скорости и
сходимости были смешаны с различием задачи. Оба исхода предотвращают ложные
выводы в основной экспериментальной главе.

## 4. Методы и профили

| ID | Реализация | Назначение |
| --- | --- | --- |
| `TF-REF` | исходный TensorFlow `article_pinn` | эталон формул и исходной скорости |
| `TORCH-PARITY` | PyTorch VP без ускоряющих изменений | проверка outputs/derivatives/loss/update |
| `TORCH-EAGER` | специализированный eager train step | эффект удаления framework overhead |
| `TORCH-COMPILED` | максимально целостный поддерживаемый compiled loss/step | измерение реального выигрыша compilation |

Primary independent variable на correctness-этапе -- backend. На performance-
этапе -- execution profile. Формулировка VP и математическая задача неизменны.

## 5. Основной датасет

Используется reference DNS `RBC_PTV_1E6_07`, Ra=`1e6`, Pr=`0.7`, 11 snapshots.
Каждый raw NPZ содержит строки:

```text
t, x, y, z, u, v, w, T, p
```

Штатный reference preprocessing объединяет snapshots `00000..00010` в:

```text
RBC_PTV_1E6_07_t_11.npz
inputs:  (t,x,y,z)
outputs: (u,v,w,T,p)
```

Этот NPZ является primary dataset для EXP-001, потому что непосредственно
соответствует TensorFlow-коду и содержит T,p для evaluation. В training data
loss используются только `u,v,w`; T,p остаются скрытой ground truth.

`all_points_10.csv` в EXP-001 не используется: его прежний pipeline дополнительно
перенормировал время. `all_points_001_metric.csv` и `all_points_005_metric.csv`
не используются, поскольку не имеют T,p ground truth и предназначены для
последующего PTV transfer.

До запуска в `docs/data_spec.md` должны быть записаны fingerprint объединенного
NPZ, shapes, dtype, ranges и подтвержденное отсутствие дополнительного time
scaling.

## 6. Каноническая VP-задача

Вход сети:

```text
(t,x,y,z)
```

Выход:

```text
(u,v,w,T,p)
```

Уравнения:

```text
r_u = u_t + u*u_x + v*u_y + w*u_z + p_x
      - sqrt(Pr/Ra)*(u_xx + u_yy + u_zz)

r_v = v_t + u*v_x + v*v_y + w*v_z + p_y
      - sqrt(Pr/Ra)*(v_xx + v_yy + v_zz)

r_w = w_t + u*w_x + v*w_y + w*w_z + p_z
      - sqrt(Pr/Ra)*(w_xx + w_yy + w_zz) - T

r_T = T_t + u*T_x + v*T_y + w*T_z
      - sqrt(1/(Pr*Ra))*(T_xx + T_yy + T_zz)

r_div = u_x + v_y + w_z
```

На parity-профиле `Nu=0`, поэтому выход сети является самой температурой без
дополнительного mean-profile transform.

## 7. Reference model и losses

Model:

```text
input_dim: 4
hidden_layers: 10 x 256
activation: sin in every hidden layer
output_dim: 5
kernel initialization: Keras Dense default Glorot uniform
bias initialization: zeros
dtype: float32 for initial parity unless reference proves otherwise
```

Loss weights из reference config:

```text
data:       1.0
NSE:        1e-1
energy:     1e-2
continuity: 1e-3
pressure center: 0
global boundary: 1e-4
active boundary terms: T(z=0)=0.5, T(z=1)=-0.5
```

Все reductions воспроизводятся буквально. `loss_NSE` усредняет квадрат трех
momentum components совместно; data loss усредняет `u,v,w` совместно.

Reference-код строит и некоторые boundary terms с нулевыми весами. Поэтому:

- `TORCH-PARITY` сначала воспроизводит observed scalar loss, а fixture явно
  хранит все поданные boundary points;
- после correctness gate `TORCH-EAGER` вычисляет только активные terms;
- выигрыш zero-weight pruning измеряется отдельно и не маскируется как backend.

## 8. Sampling и batch semantics

Reference batch size: `4096`.

Для каждого data minibatch:

- времена collocation points берутся из текущих data times;
- `x,y,z` domain points выбираются uniform в `[0,1]^3`;
- boundary points используют те же batch times;
- выполняется один total loss, один backward и один Adam update.

Для correctness parity случайная генерация отключается: fixture содержит явно
сохраненные data/domain/boundary tensors. После parity performance-профиль снова
использует sampling внутри train step.

Keras shuffle order и PyTorch sampler order задаются явно. Понятие epoch не
используется для cross-backend performance conclusion.

## 9. Перенос весов между backend

Создается deterministic weight fixture. Для каждого Dense/Linear слоя:

```text
torch_weight = transpose(tf_kernel)
torch_bias = tf_bias
```

Имена слоев и mapping сохраняются в manifest. Сначала сравнивается forward,
затем производные и только после этого parameter gradients/update.

## 10. Оптимизация

Reference:

```text
Adam(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-7)
ReduceLROnPlateau(factor=0.8, patience=100, min_lr=1e-4, min_delta=5e-6)
```

Для one-step parity scheduler не участвует. Adam states и step index начинаются
одинаково. PyTorch `fused` сначала выключен. После parity `fused=True` является
отдельным performance-профилем с проверкой результата.

Полный training benchmark сравнивает одинаковое число optimizer updates и одну
LR policy. Scheduler обновляется в одной и той же логической единице; backend
defaults запрещены.

## 11. Порядок correctness checks

На небольшом фиксированном fixture:

1. mapped parameters;
2. model outputs на data/domain/boundary points;
3. первые производные `u,v,w,T,p`;
4. требуемые diagonal second derivatives;
5. `r_u,r_v,r_w,r_T,r_div`;
6. каждый reduced loss component;
7. total scalar loss;
8. gradients total loss по каждому parameter tensor;
9. один Adam update и новое значение параметров.

Сравнения выполняются через max absolute error, relative L2 и `allclose` с явно
заданными `atol/rtol`. Начальные proposed tolerances для float32:

```text
outputs/first derivatives: atol=1e-5, rtol=1e-4
second derivatives/residuals: atol=5e-5, rtol=5e-4
losses/parameter gradients: atol=1e-4, rtol=1e-3
```

Это стартовые критерии, не право расширять tolerance до прохождения теста. Любое
изменение tolerance документируется вместе с причиной и распределением ошибки.

## 12. Performance protocol

После correctness gate:

- одна и та же GPU model;
- одинаковые batch/point counts;
- отдельные warm-up и compile phases;
- не менее 100 steady-state steps для микробенчмарка;
- синхронизация GPU вокруг измеряемых интервалов;
- минимум 5 повторов процесса benchmark;
- logging, checkpoint и full evaluation выключены в microbenchmark;
- end-to-end benchmark дополнительно включает заданную production logging policy.

Измеряется:

```text
sampling_ms
data_forward_ms
physics_forward_derivatives_ms
boundary_ms
loss_assembly_ms
backward_ms
optimizer_ms
total_step_ms
peak_allocated_memory
peak_reserved_memory
```

Отдельно измеряются:

- zero-weight branch pruning;
- eager против compiled;
- обычный против fused Adam;
- стоимость W&B logging/evaluation вне hot step.

## 13. W&B и DataSphere

W&B:

```text
project: diploma-pinn
group: EXP-001
job_type: parity | benchmark
```

Графики строятся по `optimizer_step` и `wall_time_seconds`. Full-field DNS
evaluation выполняется редко и не входит в microbenchmark.

Планируемые DataSphere configs:

```text
datasphere/exp-001-tf-reference.yaml
datasphere/exp-001-torch-parity.yaml
datasphere/exp-001-torch-benchmark.yaml
```

Они создаются после реализации CLI и environment lock. Каждый job проверяет
dataset fingerprint до запуска и выгружает manifest, resolved config, summary и
profiler results. Полный контракт описан в `docs/external_services.md`.

## 14. Acceptance и stop criteria

### Gate 1: formulation parity

Все outputs, derivatives, residuals и loss components проходят tolerance либо
каждое оставшееся расхождение локализовано и доказано как backend-specific.

### Gate 2: update parity

Parameter gradients совпадают. Adam update совпадает либо отличие полностью
объяснено конкретной формулой optimizer, после чего выбирается один явно
задокументированный parity optimizer profile.

### Gate 3: performance readiness

Измерение устойчиво между повторами, point counts совпадают, profiler accounting
покрывает весь step, compiled/eager correctness проверена.

### Stop

Не запускать длинное обучение, если расходятся residuals/losses, датасет не прошел
fingerprint/schema checks, PyTorch silently использует T,p labels или step timing
содержит full evaluation/checkpoint overhead.

## 15. Ожидаемые артефакты

```text
tests/fixtures/exp001/
  points_and_weights.*
  expected_outputs_and_residuals.*

configs/experiments/exp001/
  tf_reference.yaml
  torch_parity.yaml
  torch_eager.yaml
  torch_compiled.yaml

outputs/EXP-001/<run_id>/
  manifest.json
  resolved_config.yaml
  parity_report.json
  benchmark_summary.json
  profiler/
```

Малые fixtures и агрегированные reports разрешено хранить в Git. Большие datasets,
checkpoints, raw profiler traces и W&B directories не коммитятся.

## 16. Результаты

Не запускалось. Раздел заполняется без изменения исходных гипотез и критериев.

## 17. Решение после EXP-001

После прохождения gates `TORCH-EAGER` или корректный `TORCH-COMPILED` становится
единым VP baseline нового проекта. Только после этого создается EXP-002 для
реализации/пилота VV и FO.

