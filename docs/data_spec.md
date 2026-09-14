# Спецификация данных

Статус: начальный контракт; значения нормировки и provenance должны быть
подтверждены до первого полного запуска.

## Источники

Старый workspace:

```text
C:/Users/egorg/PYTHON_WORK/PINN/multipinnFolder
```

| Набор | Роль | Допустимое использование |
| --- | --- | --- |
| `all_points_10.csv` | DNS с `u,v,w,T,p` | `u,v,w` как train observations; `T,p` скрыты и используются для evaluation |
| `all_points_001_metric.csv` | разреженные PTV-derived `u,v,w`; T,p placeholders | velocity-only reconstruction |
| `all_points_005_metric.csv` | более плотные PTV-derived `u,v,w`; T,p placeholders | velocity-only reconstruction |
| raw `0.01ppp` / `0.05ppp` | координаты, время/кадр, `PartID` | provenance и возможные будущие trajectory studies, не текущий baseline |
| `article_pinn/RBC-PINN` data | TensorFlow reference case | parity и внешний baseline |

## Приоритет наборов

1. **Primary development/evaluation dataset:** штатный DNS reference
   `RBC_PTV_1E6_07_t_11.npz`, собранный без дополнительного time scaling из 11
   snapshots. Он используется в EXP-001 и следующих контролируемых DNS опытах.
   Проверенный паспорт: [`datasets/RBC_PTV_1E6_07.md`](datasets/RBC_PTV_1E6_07.md).
2. **Secondary transfer datasets:** `all_points_001_metric.csv` и
   `all_points_005_metric.csv`. Они используются после валидации метода для
   восстановления неизвестных T,p из velocity-only PTV data.
3. `all_points_10.csv` не является primary, пока не разрешено обнаруженное
   дополнительное масштабирование времени относительно PDE coefficients.

Наличие T,p в primary DNS удобно не потому, что они подаются сети, а потому, что
после velocity-only обучения позволяют немедленно и количественно проверить
реконструкцию скрытых полей.

Публичный `RBC_PTV_1E6_07` имеет `PTV` в имени, но фактически содержит 11 полей
на одной неподвижной сетке `64^3`, а не particle tracks. Использовать термин
«лагранжевы наблюдения» для этого конкретного test case нельзя.

Большие файлы не копируются и не коммитятся. Локальные пути задаются через
игнорируемый `configs/local/` или переменные окружения.

## Канонические схемы

Observation batch:

```text
points: float tensor [N,4], columns (t,x,y,z)
velocity: float tensor [N,3], columns (u,v,w)
optional trajectory_id: integer tensor [N]
```

Evaluation batch DNS:

```text
points: [N,4]
fields: [N,5], named (u,v,w,T,p)
```

Наличие столбцов не означает разрешение использовать их в train loss. Такое
разрешение задается только experiment spec.

## Обязательные проверки до обучения

- provenance и физический режим каждого набора;
- единицы и nondimensionalization;
- порядок координат и направление gravity;
- Ra, Pr и коэффициенты PDE;
- диапазоны координат и времени;
- обнаруженное в старом pipeline преобразование времени `[0,0.5] -> [0,1]`;
- отсутствие T,p leakage в training loader;
- split по целым временам/областям, а не случайным соседним строкам;
- pressure gauge в evaluation;
- fingerprint исходного файла и версия preprocessing.

## Split policy

Каждый split получает стабильный идентификатор и отдельный manifest. Для DNS
training видит только разрешенные velocity rows. T,p остаются в evaluator.

Для PTV нельзя считать нулевые T,p ground truth. Без независимой температуры или
давления выводы ограничиваются скоростью, PDE consistency и физическими
статистиками.
