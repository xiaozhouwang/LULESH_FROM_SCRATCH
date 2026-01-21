# LULESH 2.0（CPU + GPU 验证 + 基准测试）

本仓库包含 CUDA GPU 版本、用于 CPU/GPU 验证的日志钩子，以及用于加速比与正确性绘图的基准工具。

**要点**
- GPU 版本位于 `lulesh-cuda`，生成 `lulesh_cuda`。
- 日志与对比工具位于 `benchmarks/`。
- 支持按周期输出 CSV 并基于容差比较的 CPU/GPU 正确性检查。
- 可从采集结果生成加速比与正确性图。

## 构建

CPU（串行/OpenMP，无 MPI）：
```
make -j CXX="g++ -DUSE_MPI=0"
```

GPU：
```
cd lulesh-cuda
./build.sh
```

## 日志与正确性验证

关键工具：
- `benchmarks/compare_logs.py`：对 CPU/GPU CSV 日志进行容差比较。
- `benchmarks/run_multi_compare.sh`：运行 CPU + GPU 并对比日志。
- `benchmarks/plot_correctness.py`：按周期汇总差异并绘图。

示例抽样正确性运行（size 110, 10 cycles, stride 100, fields fx/fy/fz/e）：
```
LULESH_LOG_STRIDE=100 LOG_CYCLES=10 LOG_CYCLE_STRIDE=1 SIZE=110 ITERS=10 \
LOG_FIELDS=fx,fy,fz,e LOG_PRE=1 LOG_SUBSTEPS=0 \
benchmarks/run_multi_compare.sh

python3 benchmarks/plot_correctness.py --plot \
  --cpu benchmarks/logs-multi/cycles10-s1 \
  --gpu benchmarks/logs-gpu-multi/cycles10-s1 \
  --allow-missing
```

## 基准与加速比图

加速比批量脚本：
```
python3 benchmarks/bench_speedup.py --sizes 30,50,70,90,110 \
  --iterations 100 --cpu-threads 24 --plot
```

输出：
- `benchmarks/speedup.csv`
- `benchmarks/speedup.png`

注意：LULESH 的 elapsed time 只保留 2 位有效数字，且只计主循环时间（GPU 不包含初始化与拷贝），建议用更大的规模或更多迭代来获得更稳定的加速比，或使用外部计时。

## 当前运行记录结果

以下为本机样例结果；请按你的硬件与设置重新生成。

硬件（本机）：
- CPU: Intel(R) Core(TM) i9-14900K（24 核，32 线程）
- GPU: NVIDIA GeForce RTX 4090（49140 MiB，计算能力 8.9，驱动 580.82.07）

示例正确性（size 110 抽样运行, 10 cycles, stride 100, fields fx/fy/fz/e）：
- Max abs diff: 5.96e-08 (cycle 6)
- Max rel diff: 3.28e-13 (cycle 4)
- Out-of-bounds count: 0 across all cycles
- Plot: `benchmarks/correctness.png`

多周期正确性（size 40, 10 cycles, stride 1）：
- 1290 files compared, 0 failures
- Max abs diff: 4.77e-07 (cycle 4)
- Max rel diff: 1.0 (near-zero values)
- Plot: `benchmarks/analysis/correctness_multi_c10_s40.png`
- Logs: `benchmarks/logs-multi/cycles10-s1` and
  `benchmarks/logs-gpu-multi/cycles10-s1`

示例加速比批量结果（CPU threads=24, iterations=100）：
- N=30: ~37.04x
- N=50: ~29.17x
- N=70: ~27.78x
- N=90: ~30.56x
- N=110: ~33.33x
- Plot: `benchmarks/speedup.png`

示例基线（size 110, iterations 100）：
- CPU: ~21.0s, FOM ~6467.8
- GPU: ~0.63s, FOM ~212521.4
- Speedup: ~33.3x

## 原始 README（原文）

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
