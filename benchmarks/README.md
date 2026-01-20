# LULESH Logging and Comparison

This folder contains helper scripts and example outputs for CPU/GPU
comparison logging.

## Generate CPU reference logs

Recommended for deterministic reference:

```
OMP_NUM_THREADS=1 LULESH_LOG_ENABLE=1 LULESH_LOG_PRE=1 ./lulesh2.0 -s 20 -i 2
```

Logging controls:

- `LULESH_LOG_ENABLE=1` enables logging.
- `LULESH_LOG_PRE=1` adds a pre-LagrangeNodal snapshot.
- `LULESH_LOG_CYCLES=...` logs cycles 1..N (default 1).
- `LULESH_LOG_CYCLE_STRIDE=...` logs every Nth cycle within 1..N (default 1).
- `LULESH_LOG_CYCLE_LIST=1,2,5` logs only the listed cycles (overrides cycle range).
- `LULESH_LOG_SUBSTEPS=1` logs intermediate nodal sub-steps.
- `LULESH_LOG_ROOT=...` overrides the log root (default `benchmarks/logs`).
- `LULESH_LOG_STRIDE=...` logs every Nth value.
- `LULESH_LOG_FIELDS=x,y,z,...` logs only selected fields.

## Compare CPU vs GPU logs

```
python3 benchmarks/compare_logs.py \
  --cpu benchmarks/logs \
  --gpu /path/to/gpu/logs \
  --precision double
```

Override tolerances if needed:

```
python3 benchmarks/compare_logs.py --cpu benchmarks/logs --gpu gpu/logs \
  --abs-tol 1e-10 --rel-tol 1e-8
```
