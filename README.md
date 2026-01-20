# LULESH 2.0 (CPU + GPU Validation + Benchmarks)

This repo now includes a CUDA GPU port, logging hooks for CPU/GPU validation,
and benchmark utilities for speedup and correctness plots.

**Highlights**
- GPU port lives under `lulesh-gpu-opt/lulesh-cuda` and builds `lulesh_gpu`.
- Logging and comparison tooling lives under `benchmarks/`.
- CPU/GPU correctness checks are supported with per-cycle CSV logs and a
  tolerance-based comparator.
- Speedup and correctness plots are generated from the collected results.

## Build

CPU (serial/OpenMP, no MPI):
```
make -j CXX="g++ -DUSE_MPI=0"
```

GPU:
```
cd lulesh-gpu-opt/lulesh-cuda
nvcc -std=c++14 -O3 -arch=sm_89 \
  lulesh.cu lulesh-util.cu lulesh-viz.cu lulesh-init.cu \
  -o lulesh_gpu
```

## Logging and Correctness Validation

Key helpers:
- `benchmarks/compare_logs.py` compares CPU/GPU CSV logs with tolerances.
- `benchmarks/run_multi_compare.sh` runs CPU + GPU and compares logs.
- `benchmarks/plot_correctness.py` summarizes diffs by cycle and plots them.

Example sampled correctness run (size 110, 10 cycles, stride 100, fields
fx/fy/fz/e):
```
LULESH_LOG_STRIDE=100 LOG_CYCLES=10 LOG_CYCLE_STRIDE=1 SIZE=110 ITERS=10 \
LOG_FIELDS=fx,fy,fz,e LOG_PRE=1 LOG_SUBSTEPS=0 \
benchmarks/run_multi_compare.sh

python3 benchmarks/plot_correctness.py --plot \
  --cpu benchmarks/logs-multi/cycles10-s1 \
  --gpu benchmarks/logs-gpu-multi/cycles10-s1 \
  --allow-missing
```

## Benchmarking and Speedup Plots

Speedup sweep script:
```
python3 benchmarks/bench_speedup.py --sizes 30,50,70,90,110 \
  --iterations 100 --cpu-threads 24 --plot
```

Outputs:
- `benchmarks/speedup.csv`
- `benchmarks/speedup.png`

## Recorded Results (Current Runs)

Correctness (size 110 sampled run, 10 cycles, stride 100, fields fx/fy/fz/e):
- Max abs diff: 5.96e-08 (cycle 6)
- Max rel diff: 3.28e-13 (cycle 4)
- Out-of-bounds count: 0 across all cycles
- Plot: `benchmarks/correctness.png`

Multi-cycle correctness (20 cycles, stride 2):
- 1290 files compared, 0 failures
- Logs: `benchmarks/logs-multi/cycles20-s2` and
  `benchmarks/logs-gpu-multi/cycles20-s2`

Speedup sweep (CPU threads=24, iterations=100):
- N=30: ~6.67x
- N=50: ~7.41x
- N=70: ~9.42x
- N=90: ~11.46x
- N=110: ~11.98x
- Plot: `benchmarks/speedup.png`

Example baseline (size 110, iterations 100):
- CPU: ~20.0s, FOM ~6592.5
- GPU: ~1.69s, FOM ~78971.2
- Speedup: ~11.8x

## Original README (verbatim)

```
This is the README for LULESH 2.0

More information including LULESH 1.0 can be found at https://codesign.llnl.gov/lulesh.php

If you have any questions or problems please contact:

Ian Karlin <karlin1@llnl.gov> or
Rob Neely <neely4@llnl.gov>

Also please send any notable results to Ian Karlin <karlin1@llnl.gov> as we are still evaluating the performance of this code.

A Makefile and a CMake build system are provided.

*** Building with CMake ***

Create a build directory and run cmake. Example:

  $ mkdir build; cd build; cmake -DCMAKE_BUILD_TYPE=Release -DMPI_CXX_COMPILER=`which mpicxx` ..

CMake variables:

  CMAKE_BUILD_TYPE      "Debug", "Release", or "RelWithDebInfo"

  CMAKE_CXX_COMPILER    Path to the C++ compiler
  MPI_CXX_COMPILER      Path to the MPI C++ compiler

  WITH_MPI=On|Off       Build with MPI (Default: On)
  WITH_OPENMP=On|Off    Build with OpenMP support (Default: On)
  WITH_SILO=On|Off      Build with support for SILO. (Default: Off).
  
  SILO_DIR              Path to SILO library (only needed when WITH_SILO is "On")

*** Notable changes in LULESH 2.0 ***

Split functionality into different files
lulesh.cc - where most (all?) of the timed functionality lies
lulesh-comm.cc - MPI functionality
lulesh-init.cc - Setup code
lulesh-viz.cc  - Support for visualization option
lulesh-util.cc - Non-timed functions

The concept of "regions" was added, although every region is the same ideal gas material, and the same sedov blast wave problem is still the only problem its hardcoded to solve. Regions allow two things important to making this proxy app more representative:

Four of the LULESH routines are now performed on a region-by-region basis, making the memory access patterns non-unit stride

Artificial load imbalances can be easily introduced that could impact parallelization strategies.  
   * The load balance flag changes region assignment.  Region number is raised to the power entered for assignment probability.  Most likely regions changes with MPI process id.
   * The cost flag raises the cost of ~45% of the regions to evaluate EOS by the entered multiple.  The cost of 5% is 10x the entered
 multiple.

MPI and OpenMP were added, and coalesced into a single version of the source that can support serial builds, MPI-only, OpenMP-only, and MPI+OpenMP

Added support to write plot files using "poor mans parallel I/O" when linked with the silo library, which in turn can be read by VisIt.

Enabled variable timestep calculation by default (courant condition), which results in an additional reduction.  Also, seeded the initial timestep based on analytical equation to allow scaling to arbitrary size.  Therefore steps to solution will differ from LULESH 1.0.

Default domain (mesh) size reduced from 45^3 to 30^3

Command line options to allow for numerous test cases without needing to recompile

Performance optimizations and code cleanup uncovered during study of LULESH 1.0

Added a "Figure of Merit" calculation (elements solved per microsecond) and output in support of using LULESH 2.0 for the 2017 CORAL procurement

*** Notable changes in LULESH 2.1 ***

Minor bug fixes.
Code cleanup to add consitancy to variable names, loop indexing, memory allocation/deallocation, etc.
Destructor added to main class to clean up when code exits.


Possible Future 2.0 minor updates (other changes possible as discovered)

* Different default parameters
* Minor code performance changes and cleanupS

TODO in future versions
* Add reader for (truly) unstructured meshes, probably serial only
```
