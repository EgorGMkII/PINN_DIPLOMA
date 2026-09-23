# EXP-006: transfer to velocity-only PTV

Status: dataset adapter ready; runs pending method selection.

## Data

`all_points_001_metric.csv` has 810,000 rows, 100 times and 8,100 rows per time. `all_points_005_metric.csv` has 4,050,000 rows, 100 times and 40,500 rows per time. Read only `t,X,Y,Z,VX,VY,VZ`; the `T,p` columns are zero placeholders and must never become labels.

## Evaluation

Train the selected formulation with time-stratified batches. There is no claim of numerical T/p accuracy without independent truth. Report held-out velocity error, independent PDE residuals, boundary consistency, heat-transport statistics and variability across seeds. Keep DNS conclusions and PTV plausibility evidence in separate tables.
