# Внешние сервисы: Weights & Biases и Yandex DataSphere

Статус: обязательный инфраструктурный контракт.

## 1. Общие правила

- Локальная машина используется для разработки, unit/parity tests и коротких
  CPU/GPU smoke runs.
- Полные обучения и профилирование на GPU выполняются в Yandex DataSphere.
- W&B используется как журнал экспериментов и средство сравнения графиков, но не
  является источником истины для конфигурации.
- Источники истины: Git commit, versioned experiment spec и resolved config,
  сохраненный рядом с результатами.
- API keys, OAuth-токены и другие секреты никогда не записываются в Git, YAML,
  Markdown, run manifest или командную строку, сохраняемую в документации.

## 2. Weights & Biases

### Назначение

W&B хранит временные ряды метрик, сравнение runs, небольшие итоговые таблицы и
ссылки на артефакты. Исследовательская единица -- optimizer step, не epoch.

Рекомендуемая организация:

```text
project: diploma-pinn
group: EXP-001
job_type: parity | benchmark | train | evaluate
name: exp001-<backend>-<profile>-seed<seed>
tags: [vp, dns-ra1e6-pr07, parity]
```

Каждый run config обязан включать:

```text
experiment_id
git_commit and git_dirty
resolved_config
dataset_id, fingerprint and split_id
seed
backend and execution_profile
device, precision and library versions
model/output schema and parameter count
points per type per optimizer step
optimizer and scheduler parameters
```

### Что логировать

По `optimizer_step` с ограниченной частотой:

- total loss и все именованные компоненты;
- learning rate;
- `u,v,w,T,p` evaluation metrics;
- pressure metric после gauge alignment;
- sampling/forward/derivatives/backward/optimizer timings;
- peak allocated/reserved GPU memory;
- cumulative wall time и processed point counts;
- NaN/Inf и gradient-norm diagnostics.

Графики W&B должны позволять сравнить runs как по `optimizer_step`, так и по
`wall_time_seconds`. Метрики полного DNS evaluation вычисляются редко и вне hot
step. Не отправлять в W&B исходные CSV/NPZ, полные поля на каждом шаге или секреты.

### Артефакты

Для каждого завершенного run сохраняются небольшие:

- resolved config;
- run manifest;
- metrics summary;
- profiler summary;
- финальный checkpoint только для выбранных runs;
- выбранные field slices/таблицы, необходимые для анализа.

Если сеть недоступна, run работает в offline mode и синхронизируется позже. Код
обучения не должен падать только из-за недоступности W&B.

Секрет `WANDB_API_KEY` задается в локальном окружении или через секреты
DataSphere. В versioned config разрешены только project/group/name/tags и logging
policy.

## 3. Yandex DataSphere

### Принцип работы

Каждый тяжелый run запускается как DataSphere job из versioned YAML-конфига, как
в предыдущем проекте. Конфиг хранится в `datasphere/` и указывает:

- одну точную команду запуска;
- воспроизводимое Python-окружение;
- локальные пути к пакету и необходимым job inputs;
- список выходных директорий;
- тип GPU VM и размер working storage;
- graceful shutdown.

Планируемые имена:

```text
datasphere/exp-001-tf-reference.yaml
datasphere/exp-001-torch-parity.yaml
datasphere/exp-001-torch-benchmark.yaml
```

Конкретный YAML создается только после появления соответствующей CLI-команды и
requirements lock. Нельзя хранить конфиг с несуществующей командой как будто он
готов к запуску.

### Контракт job

Job перед обучением обязан:

1. вывести experiment ID, commit и resolved config;
2. проверить наличие и fingerprint датасета;
3. вывести GPU, CUDA, backend и версии библиотек;
4. выполнить быстрый data/model smoke check;
5. создать уникальный run directory;
6. только после этого начать тяжелое вычисление.

Ожидаемый layout результата:

```text
outputs/<experiment_id>/<run_id>/
  manifest.json
  resolved_config.yaml
  metrics.jsonl
  summary.json
  profiler/
  figures/
  checkpoints/
```

Job должен корректно завершаться по `SIGTERM`, сохранить последний разрешенный
checkpoint/summary и закрыть W&B run.

### Данные в job

Данные не хранятся в Git. Для EXP-001 в job передается подготовленный reference
NPZ или 11 исходных DNS snapshots через явно указанный локальный input path.
Первая операция -- проверка fingerprint и схемы. Путь на VM не должен быть
зашит в Python-код; он передается experiment config.

### Управление и безопасность

Проект DataSphere и auth настраиваются вне repository. В документации допустимы
project ID и безопасные CLI-шаблоны, но не IAM/OAuth token. Типовой жизненный
цикл после появления job config:

```text
datasphere project job execute -p <project_id> -c <job-config.yaml>
datasphere project job list -p <project_id>
datasphere project job download-files --id <job_id>
```

После скачивания `job_id`, DataSphere image/instance, Git commit и W&B run URL
добавляются в experiment results. Job config и resolved experiment config должны
однозначно воспроизводить запуск.

## 4. Граница ответственности

```text
experiment spec  -> задает вопрос и критерий успеха
experiment config -> задает модель, данные, points, optimizer и logging
DataSphere YAML   -> задает удаленную среду и команду
W&B               -> показывает ход и сравнение runs
run manifest      -> связывает все перечисленное с commit и dataset fingerprint
```

W&B dashboard или DataSphere job history не заменяют versioned документы и
конфиги: удаленные записи могут быть удалены или стать недоступными.

