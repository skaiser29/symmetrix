# Aurora Water Performance Pass (2026-04-27)

## Goal

Improve SymmetriX/LAMMPS Aurora performance from a fresh `develop` branch, with special attention to the weak 1-rank water benchmark path reported around `0.22 ns/day`, while preserving correctness and the existing 12-rank one-node path.

## Starting Point

- Branch: `perf/aurora-water-1r-2026-04-27`
- Base: `origin/develop` at `229b5e1` (`fix rrnlb retained buffer clear extents`)
- Worktree: `/home/skaiser/soft/symmetrix-aurora-water-perf`
- Reported target context: one-node water benchmark around `1.55 ns/day`; desired `>2 ns/day`; 1-rank around `0.22 ns/day`.

## Plan

- [x] Create a fresh branch from `origin/develop`.
- [x] Record this plan in `tasks/todo.md`.
- [x] Review Aurora machine and node performance docs for hardware-relevant constraints.
- [x] Review local SymmetriX/LAMMPS notes, branches, benchmark scripts, and development history under `/home/skaiser/soft`.
- [x] Identify the active water benchmark scripts and current build wiring.
- [x] Build an isolated LAMMPS binary wired to this SymmetriX branch.
- [x] Reject unsafe/non-equivalent benchmark modes before changing source.
- [x] Reproduce baseline 1-rank and one-node water performance with the 2025.2 develop-based source.
- [x] Benchmark existing tuning knobs to choose the smallest source change.
- [x] Implement the simplest low-risk Aurora-focused optimization candidate.
- [x] Rebuild and run correctness checks against the baseline.
- [x] Re-run 1-rank and one-node performance comparisons.
- [x] Review whether the result is elegant enough for a staff-engineer review.
- [x] Document final results, residual risks, and follow-up opportunities.

## Early Hardware Notes

- Aurora nodes expose 12 GPU tiles as the natural full-node GPU unit.
- The ALCF node benchmark page reports roughly 1 TB/s memory triad bandwidth per tile and 12 TB/s full-node, with similarly near-12x full-node scaling for FP64/FP32 GEMM.
- If 12-rank performance is acceptable but 1-rank is poor, likely candidates include fixed overheads, under-occupied kernels, unnecessary launches, host/device synchronization, small GEMMs, or work partitioning that only amortizes at full-node scale.

## Local History Notes

- Active prior 770-atom Aurora water benchmark context is `/home/skaiser/tests/water_bench_suite`.
- Prior stable numbers are approximately `1r=0.219-0.225 ns/day` and `12r=1.56-1.58 ns/day`.
- Prior profiling showed the 1-rank case is pair/RRNLB-kernel bound, with `compute_rrnlb_interaction_layer_forward` and `reverse_rrnlb_interaction_layer` dominating; oneMKL GEMM and sphericart were not the leading costs.
- Previous forward-staging work improved 1-rank only modestly and left 12-rank roughly flat; a receiver-owned reverse rewrite regressed badly and should not be repeated.
- The benchmark harness currently forces `pairmode=mpi_message_passing`; the pair style itself defaults to `no_domain_decomposition` for a single MPI rank when no mode is supplied. A 1-rank A/B of those modes is required before changing source.
- `no_domain_decomposition` is not a valid shortcut for this benchmark: the initial 1-rank run diverged quickly and then faulted on the GPU, so optimization work must preserve the MPI-message-passing path used by the stable harness.

## Build Notes

- SymmetriX worktree: `/home/skaiser/soft/symmetrix-aurora-water-perf`.
- LAMMPS build worktree: `/home/skaiser/soft/lammps-aurora-water-perf`.
- The LAMMPS pair symlinks and CMake cache are wired to this SymmetriX worktree, not `/home/skaiser/soft/symmetrix-aurora-stable`.
- Built `/home/skaiser/soft/lammps-aurora-water-perf/build_lean/lmp` first with the current default 2025.3 stack; that binary was much slower than historical Aurora results and is not a fair optimization baseline.
- Rebuilt `/home/skaiser/soft/lammps-aurora-water-perf/build_lean_25190/lmp` with `/opt/aurora/25.190.0` modules, `oneapi/release/2025.2.0`, and `mpich/opt/develop-git.6037a7a`, matching the historical benchmark stack.
- Build cache confirms `LibSymmetrix_SOURCE_DIR=/home/skaiser/soft/symmetrix-aurora-water-perf/libsymmetrix`, `SYMMETRIX_KOKKOS=ON`, `CMAKE_BUILD_TYPE=Release`, `Kokkos_ARCH_INTEL_PVC=ON`, and lean KokkosKernels batched/BLAS-only components.
- The cache exposes `SYMMETRIX_SPHERICART_SYCL=ON`, but the current SymmetriX `compute_Y` path still uses the host sphericart implementation outside CUDA; prior VTune notes show this is not the leading 1-rank cost.
- `lmp -h` cannot run on the login node because MPI initialization aborts there, so execution verification must happen under PBS.

## Benchmark Matrix

- Baseline: `RANKS=1,12`, `PAIRMODE=mpi_message_passing`, `RUNSTEPS=1000`.
- Unsafe mode A/B was removed after `no_domain_decomposition` diverged; remaining 1-rank experiments use `PAIRMODE=mpi_message_passing`.
- Low-risk policy A/B after baseline: force split forward for the 770-atom 1-rank case via existing env knobs before hardcoding any policy change.
- Vector-length A/B uses existing `SYMMETRIX_RRNLB_FORWARD_VECTOR_LENGTH` and `SYMMETRIX_RRNLB_REVERSE_VECTOR_LENGTH` knobs.
- Correctness: compare thermo energy/drift and net-force behavior between baseline and candidate runs, not speed alone.

## Submitted Runs

- `8452999.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`: one-node debug matrix via `tasks/run_aurora_water_perf_matrix.pbs`, `RUNSTEPS=1000`.
- Matrix cases:
  - `baseline_mpi`: `RANKS=1,12`, `PAIRMODE=mpi_message_passing`.
  - `r1_no_domain`: `RANKS=1`, `PAIRMODE=no_domain_decomposition`.
  - `r1_force_split`: existing env `SYMMETRIX_RRNLB_FORWARD_ADAPTIVE_MODE=force_split`.
  - `r1_vec8`: existing env vector length knobs set to `8`.
  - `r1_force_split_vec8`: combined force-split plus vector length `8`.
- Result: the 2025.3-stack baseline was only `1r=0.062 ns/day`, `12r=0.427 ns/day`, far below the historical `1r~0.22`, `12r~1.55` reference. This points to toolchain/runtime stack sensitivity rather than a useful source-level comparison.
- Result: `r1_no_domain` showed nonphysical thermo blow-up and then a GPU segmentation fault, so it is excluded from future candidate fixes.

## Resumed Run Plan (2026-04-28)

- Use `/home/skaiser/soft/lammps-aurora-water-perf/build_lean_25190/lmp` for all benchmark comparisons.
- Run the cleaned matrix: baseline `1,12` ranks plus 1-rank force-split/vector-length candidates, all with `PAIRMODE=mpi_message_passing`.
- Choose the source change only after the matrix identifies a candidate that preserves drift and improves timing.
- Submitted `8453215.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov` with the 2025.2-stack binary and `RUNSTEPS=1000`.
- `8453215` failed before LAMMPS launched because `module purge` removed `/opt/cray/pals/1.8/bin` from `PATH`, so the harness could not find `mpiexec`.
- Patched the matrix script to restore the PALS launcher path after loading the 2025.2 stack.
- Resubmitted as `8453222.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- `8453222` launched correctly, but the run confirmed the develop-based source was still on the slow portable RRNLB linear path. The build cache had `KokkosKernels_ENABLE_TPL_MKL=OFF`, and the source lacked `SYMMETRIX_HAVE_ONEMKL`/`oneapi::mkl::blas::gemm_batch` calls.
- Canceled `8453222` after collecting that evidence; candidate knob comparisons on the portable backend would not answer the Aurora production-performance question.
- Applied the narrow known fast-path change from local commit `f764f80` to restore SYCL oneMKL RRNLB linear calls and CMake linkage on this branch.
- Reconfigured/rebuilt `/home/skaiser/soft/lammps-aurora-water-perf/build_lean_25190/lmp`; `liblammps.so` now links `libmkl_sycl_blas.so.5`, imports `oneapi::mkl::blas::row_major::gemm_batch`, and `flags.make` contains `-DSYMMETRIX_HAVE_ONEMKL`.
- Submitted 100-step smoke matrix as `8453247.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- 100-step smoke results:
  - baseline auto: `1r=0.218 ns/day`, `12r=1.570 ns/day`, stable thermo/net-force.
  - forced split-forward: `1r=0.319 ns/day`, stable thermo/net-force.
  - vector length `8`: invalid; energy was nonphysical at the first thermo line and the GPU faulted on an atomic access.
- Patched auto policy so SYCL + oneMKL builds use the existing split receiver-forward path by default while preserving explicit `force_fused` and CUDA/portable behavior.
- Rebuilt after the policy patch; `lmp` and `liblammps.so` relinked successfully.
- Revised the final validation matrix to remove invalid vector-length cases and compare patched auto against `force_fused`/`force_split` controls.
- Submitted 1000-step validation matrix as `8453268.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.

## Verification Requirements

- Use matched old-vs-new inputs, seeds, rank counts, and output precision.
- Compare thermo/energy/force behavior, not only speed.
- Compare 1-rank and one-node results separately.
- Keep source changes narrow and directly tied to measured bottlenecks.

## Final Validation

- Branch pushed: `origin/perf/aurora-water-1r-2026-04-27`.
- Commit: `c3195fa` (`aurora: route RRNLB linears through oneMKL`).
- Final validation job: `8453268.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- Binary: `/home/skaiser/soft/lammps-aurora-water-perf/build_lean_25190/lmp`.
- All final cases used `PAIRMODE=mpi_message_passing`, `RUNSTEPS=1000`, and the 2025.2 Aurora module stack.
- Error scan found no `Segmentation fault`, `Abort`, `ERROR`, `NaN`, non-numeric thermo, GPU page fault, or traceback entries in the final validation output.

| case | ranks | speed ns/day | drift meV/atom/ps fit | drift stderr | delta K | min nonzero dE eV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| patched auto | 1 | 0.319 | 0.000284318214 | 0.008452450751 | 7.67004700 | 1.384899951518e-04 |
| patched auto | 12 | 1.576 | 0.000308818085 | 0.008452582070 | 7.66945255 | 1.312490785494e-04 |
| forced fused control | 1 | 0.218 | 0.000336993672 | 0.008452676535 | 7.66997596 | 1.327750505880e-04 |
| forced split control | 1 | 0.319 | 0.000339029134 | 0.008453029888 | 7.67092872 | 1.363427145407e-04 |

## Review

- The accepted source change restores the SYCL oneMKL batched-GEMM RRNLB linear fast path and changes only the auto policy for SYCL + oneMKL builds so they choose the existing split receiver-forward path by default.
- Explicit environment overrides still work: `force_fused` keeps the old fused behavior and `force_split` matches the new auto behavior.
- CUDA and portable non-oneMKL behavior stay on the prior node-count policy.
- Rank-1 speed improved from `0.218` to `0.319 ns/day` on the validated water benchmark, about a 46% gain.
- Rank-12 one-node speed stayed at the historical level, `1.576 ns/day`, so this pass improves the anemic rank-1 path but does not reach the requested `>2 ns/day` one-node target.
- The vector-length knob is unsafe for this benchmark and was rejected after nonphysical first-step energy and a GPU fault.
- `no_domain_decomposition` is unsafe for this benchmark and was rejected after thermo blow-up and a GPU fault.
- The next likely performance target is not another policy toggle; it needs a deeper reverse/RRNLB kernel or communication/work-partitioning change that preserves the validated MPI-message-passing physics.

# Aurora 12-Rank Transfer Pass (2026-04-28)

## Goal

Explain why the roughly 46% rank-1 improvement did not transfer to the 12-rank one-node benchmark, determine whether the rank-1 win is a legitimate stack improvement or a narrow rank-1 policy win, and identify the next optimization that can move the full-node case.

## Plan

- [x] Create a follow-up branch from the validated rank-1 improvement branch.
- [x] Explain the rank-1 speedup mechanism from the RRNLB source paths.
- [x] Compare final rank-1 and rank-12 benchmark logs/timers.
- [x] Check whether rank-12 is already on the improved path or blocked by a different bottleneck.
- [x] Run a focused rank-12 A/B if existing logs are insufficient.
- [ ] Pick the smallest full-node optimization candidate with a correctness validation plan.
- [x] Record conclusions and any branch changes.

## Branch

- Branch: `perf/aurora-water-12r-transfer-2026-04-28`
- Base: `c3195fa` (`aurora: route RRNLB linears through oneMKL`)

## Findings

- The rank-1 gain was legitimate, but narrow: it changed rank 1 from the fused receiver-forward kernel to the split receiver-forward path, where `linear_1` and `linear_2` use the SYCL oneMKL batched-GEMM implementation.
- The old auto policy split only when local `num_nodes < 384`; rank 1 has about `770` local nodes, so it stayed fused before the patch.
- Rank 12 has about `64` local nodes per rank (`58-70` range), so it was already below the old split threshold and already on the split/oneMKL path before the rank-1 policy patch.
- Final validation job `8453268` showed rank 1 auto equals forced split (`0.319 ns/day`) and not forced fused (`0.218 ns/day`), proving the mechanism.
- Focused transfer job `8453301` showed rank 12 auto and forced split are the same within noise (`1.568` and `1.575 ns/day`), while forced fused regresses to `1.217 ns/day`.
- Restored the local phase CSV emitter for diagnostics. Phase CSVs from `8453301` show both rank 1 and rank 12 auto execute `forward_split_calls=2` per step and `forward_full_fused_calls=0`.
- Rank 12 therefore did not miss the rank-1 optimization; the optimization was already present in the rank-12 baseline.
- The full-node ceiling is now mostly remaining pair compute plus nontrivial MPI/halo overhead. In `8453301`, rank-12 split has `Pair avg=25.191 s` and `Comm avg=2.043 s` over 1000 steps. Zeroing communication entirely would still not reach `2 ns/day`; the pair path must move.

## Transfer Benchmark

| case | ranks | speed ns/day | loop seconds | pair avg seconds | comm avg seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| auto | 12 | 1.568 | 27.5565 | 25.265 | 2.093 |
| force fused | 12 | 1.217 | 35.5047 | 33.262 | 2.054 |
| force split | 12 | 1.575 | 27.4275 | 25.191 | 2.043 |

## Current Interpretation

- The prior commit is not a hacky wrong-physics shortcut; it selects the better existing split oneMKL execution strategy and validated cleanly for drift and energy.
- It is rank-count-sensitive because the old policy was rank-count-sensitive through local `num_nodes`; the 12-rank case already enjoyed the same split strategy.
- The next optimization should target work still shared by 1-rank and 12-rank split paths, especially reverse/RRNLB/Phi-style kernels and pair-side halo/pack/unpack behavior. A forward-only split-policy change cannot yield another 50% full-node gain.

# 5184-Atom Water Profiling And Reverse-Path Planning (2026-04-28)

## Goal

Profile the corrected 5184-atom RRNLB water workload on Aurora at 1 rank and 12 ranks, compare scaling against the 770-atom case, and use the larger-workload shape to plan a more efficient reverse path without overfitting to tiny local workloads.

## Plan

- [x] Locate the corrected 5184-atom RRNLB Aurora benchmark harness.
- [x] Confirm the workload uses `in.water_5184_rrnlb_aurora.lmp` and `/home/skaiser/tests/mace-mh-1-1-8-17-omol.json`.
- [x] Run matched 1-rank and 12-rank 5184-atom speed tests.
- [x] Run phase-CSV profiling for 1-rank and 12-rank 5184-atom runs.
- [x] If runtime permits, collect short VTune GPU hotspots for 1-rank and 12-rank.
- [x] Compare speedup, pair/comm timers, local/ghost counts, and phase counters.
- [x] Decide whether 12-rank scaling is limited primarily by local work size, reverse kernels, communication, or synchronization.
- [x] Draft a reverse-path optimization plan with correctness and performance gates across both 770-atom and 5184-atom workloads.

## Harness

- Benchmark root: `/home/skaiser/tests/water_benchmark_5184_symmetrix`.
- Data: `/home/skaiser/tests/water_benchmark_5184_symmetrix/inputs/seed192_5184.data`.
- Template: `/home/skaiser/tests/water_benchmark_5184_symmetrix/scripts/in.water_5184_rrnlb_aurora.lmp`.
- Model: `/home/skaiser/tests/mace-mh-1-1-8-17-omol.json`.
- Binary: `/home/skaiser/soft/lammps-aurora-water-perf/build_lean_25190/lmp`.

## Submitted Runs

- Speed and phase job: `8453324.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- VTune job: `8453344.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- Output root: `/home/skaiser/tests/water_benchmark_5184_symmetrix/runs`.
- Both jobs used the 2025.2 Aurora stack and `/home/skaiser/soft/lammps-aurora-water-perf/build_lean_25190/lmp`.
- Error scan found no `Segmentation fault`, `Abort`, `ERROR`, `NaN`, non-numeric thermo, GPU page fault, or traceback entries in the 1000-step speed/phase output.

## 5184-Atom Speed Results

The larger case scales much better than the 770-atom case: `0.053 -> 0.477 ns/day`, or about `9.0x` speedup on 12 ranks. Pair time scales similarly, `818.81 -> 88.06 s`, or about `9.3x`. Parallel efficiency is therefore around `75%`, not ideal but not the severe tiny-workload limit seen for 770 atoms.

| ranks | speed ns/day | loop seconds | pair avg seconds | comm avg seconds | Nlocal avg | Nghost avg |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.053 | 820.251 | 818.810 | 1.244 | 5184.00 | 9803.00 |
| 12 | 0.477 | 90.5065 | 88.060 | 2.233 | 432.00 | 2953.67 |

Correctness on the 1000-step speed run stayed matched between 1 rank and 12 ranks:

| ranks | drift meV/atom/ps fit | drift stderr | delta K | min nonzero dE eV |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.001467878038 | 0.005637335880 | 2.00890142 | 2.586971968412e-04 |
| 12 | 0.001461251282 | 0.005637371881 | 2.00862921 | 2.560685388744e-04 |

## Phase Findings

- Both 1 rank and 12 ranks execute `forward_split_calls=2` per step and `forward_full_fused_calls=0`.
- Both execute `fused_reverse_global_staged_calls=2` per step and `fused_reverse_scratch_tiled_calls=0`.
- Both use auto forward policy: `forward_adaptive_mode_auto_calls=2`, with no force-fused or force-split override.
- The current phase CSV only has useful timing for MPI pack/unpack; the compute phase timing fields are present but not wired yet.
- Average comm pack/unpack from the phase CSV:
  - 1 rank: pack `0.647 ms/step`, unpack `0.634 ms/step`.
  - 12 ranks: pack `0.266 ms/rank-step`, unpack `0.257 ms/rank-step`.

## VTune Findings

VTune reports are short 30-step profiles, so use them for hotspot mix, not absolute speed. Kernel time is aggregated over reported GPU tasks.

| rank case | total kernel seconds | top family | top family share | next families |
| --- | ---: | --- | ---: | --- |
| 1 rank | 23.715 | `reverse_rrnlb_interaction_layer` | 57.39% | `compute_rrnlb_interaction_layer_forward` 12.34%, `oneMKL_gemm` 12.05%, `reverse_M0_mixed_rrnlb` 5.32% |
| 12 ranks | 28.073 | `reverse_rrnlb_interaction_layer` | 50.08% | `oneMKL_gemm` 11.97%, `compute_rrnlb_interaction_layer_forward` 11.70%, `zeCommandListAppendMemoryCopy` 6.92%, `reverse_M0_mixed_rrnlb` 4.61% |

The forward split policy is not the next full-node lever. The dominant shared hotspot is `reverse_rrnlb_interaction_layer`, with the current default path in `libsymmetrix/source/mace_kokkos.cpp` using the edge-parallel reverse convolution and atomically scattering per-edge sender adjoints into `h_up_adj`.

## Interpretation

- The 5184-atom workload has enough local work to get useful domain-decomposition scaling: 12 ranks have about `432` local atoms per rank, not `64` as in the 770-atom case.
- The remaining gap to ideal 12x scaling is not primarily LAMMPS `Comm`: in the 1000-step 12-rank run, `Comm avg=2.233 s` out of `90.5065 s`, while `Pair avg=88.06 s`.
- For the 770-atom 12-rank benchmark, removing all measured LAMMPS comm would still not reach `2 ns/day`; the pair path must move.
- The large-workload VTune profile says the same thing more directly: reverse RRNLB compute is the largest shared cost at both 1 rank and 12 ranks.

## Reverse-Path Plan

- [ ] Add precise phase timing around reverse sub-stages before changing the algorithm:
  - `linear_2^T`, fused gate+normalize reverse, `linear_1^T`, `linear_res^T`, `h_up_adj` scatter, reverse convolution, `linear_up^T`, `reverse_M0_mixed_rrnlb`, `reverse_M1_mixed_rrnlb`, and MPI reverse pack/unpack fences.
  - Keep the CSV emitter behind the existing diagnostic env path so production runs are unaffected.
- [ ] Use that timing plus VTune to choose one reverse candidate at a time. The first candidate should be the reverse convolution in `reverse_rrnlb_interaction_layer`, because it is the clear top family in both 5184 profiles.
- [ ] Revisit sender-owned or sender-segmented reverse only as a controlled experiment. The code already has a disabled `edge-parallel sender-tiled` branch, but previous history warns that sender-owned reverse rewrites can regress drift or performance. Any revival must be gated by exact force/energy checks and must stay off by default until proven.
- [ ] If sender-segmented reverse is still unattractive after instrumentation, target a narrower low-risk change: reduce atomics and scratch traffic inside the existing edge-parallel reverse kernel without changing ownership semantics.
- [ ] Treat `zeCommandListAppendMemoryCopy` and oneMKL launch/copy overhead as a second-order 12-rank target after the reverse convolution. It is visible at 12 ranks, but it is much smaller than reverse RRNLB.
- [ ] Avoid rank-count special cases. Use workload/topology properties only if needed, and validate both 770 and 5184 atoms so the fix does not overfit the small benchmark.

## Validation Gates For Reverse Changes

- 770-atom benchmark: 1000-step 1-rank and 12-rank runs, no speed regression above noise, no energy/force drift regression.
- 5184-atom benchmark: 1000-step 1-rank and 12-rank runs, no drift regression, no GPU faults, and measurable 12-rank pair-time improvement.
- Compare thermo, fitted drift, net force, minimum nonzero energy delta, LAMMPS timers, phase counters, and short VTune hotspot mix against the current branch.
- A useful first acceptance bar is `>=5%` 5184 12-rank pair-time reduction with neutral 770 performance. A `2 ns/day` 770 one-node result needs a larger pair-side reduction, so this should be treated as an iterative reverse-path program rather than a single policy toggle.

## Review

- Staff-engineer checkpoint: the clean next step is instrumentation plus a measured reverse-convolution experiment, not another forward policy knob and not a rank-12-only shortcut.
- The larger workload answers the local-work question: 5184 atoms is not fundamentally too small for one-node domain decomposition, but the pair/reverse kernels still do not scale ideally.
- Current branch changes are diagnostic only: the phase CSV emitter was restored and extended. No production reverse-path change has been made yet.

# Reverse Instrumentation Pass (2026-04-28)

## Goal

Add precise, low-overhead diagnostic timing for the RRNLB reverse path, rebuild, and collect enough 770-atom and 5184-atom evidence to choose the first reverse optimization candidate without relying only on VTune kernel-family attribution.

## Plan

- [x] Inspect the current phase counter and timer plumbing.
- [x] Add gated reverse sub-stage timers and CSV columns for both RRNLB layers.
- [x] Cover these sub-stages: `linear_2^T`, fused gate+normalize reverse, `linear_1^T`, `linear_res^T`, `h_up_adj` scatter, reverse convolution, `linear_up^T`, input-adjoint scatter, `reverse_M0_mixed_rrnlb`, `reverse_M1_mixed_rrnlb`, reverse MPI comm, and layer-level totals.
- [x] Keep instrumentation off unless the existing phase CSV diagnostic environment is enabled.
- [x] Rebuild `/home/skaiser/soft/lammps-aurora-water-perf/build_lean_25190/lmp`.
- [x] Run focused phase timing for 770 atoms at 1 rank and 12 ranks.
- [x] Run focused phase timing for 5184 atoms at 1 rank and 12 ranks.
- [x] Analyze timing shares and update this file with the first optimization candidate and validation gates.

## Design Notes

- Prefer elapsed wall timing around existing coarse sub-stage calls with Kokkos fences only when diagnostics are enabled; production code should keep the existing ordering.
- Attribute reverse work by explicit stage columns rather than trying to infer it from `reverse_rrnlb_interaction_layer` alone.
- Do not alter the algorithm or default execution path during this pass.

## Implementation Notes

- Added `rrnlb_set_phase_stats_enabled()` and reverse-stage counters in `libsymmetrix/source/mace_kokkos.hpp`.
- The phase flag is enabled only when `SYMMETRIX_RRNLB_PHASE_CSV` is configured by the LAMMPS pair style.
- Added fenced timing for `reverse_rrnlb_interaction_layer` sub-stages in `libsymmetrix/source/mace_kokkos.cpp`.
- Added MPI-path outer reverse timing around product transposes, `reverse_M0/M1`, and reverse communication in `pair_symmetrix/pair_symmetrix_mace_kokkos.cpp`.
- Build succeeded for `/home/skaiser/soft/lammps-aurora-water-perf/build_lean_25190/lmp` with the 2025.2 Aurora stack.

## Timing Run

- Job: `8453381.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- Node: `x4502c7s6b0n0`.
- Output root: `/home/skaiser/tests/water_benchmark_5184_symmetrix/runs/reverse_timing_8453381.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- Script: `tasks/run_aurora_reverse_timing_matrix.pbs`.
- Cases: 770 atoms at 1 and 12 ranks for 120 steps; 5184 atoms at 1 and 12 ranks for 80 steps.
- Error scan found no `Segmentation fault`, `Abort`, `ERROR`, `NaN`, GPU page fault, or traceback entries.

| case | ranks | speed ns/day | loop seconds | pair avg seconds | comm avg seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 770 phase | 1 | 0.325 | 15.9504 | 15.891 | 0.033 |
| 770 phase | 12 | 1.550 | 3.34413 | 3.077 | 0.245 |
| 5184 phase | 1 | 0.053 | 64.7135 | 64.633 | 0.066 |
| 5184 phase | 12 | 0.499 | 6.93138 | 6.794 | 0.122 |

Short timing runs are noisier for drift than the earlier 1000-step validation runs, but matched rank-to-rank within each system:

| case | ranks | drift meV/atom/ps fit | drift stderr | delta K |
| --- | ---: | ---: | ---: | ---: |
| 770 phase | 1 | -0.122389030769 | 0.243563390622 | 27.74466247 |
| 770 phase | 12 | -0.122406373446 | 0.243555323258 | 27.74464899 |
| 5184 phase | 1 | -1.120680389859 | 0.386509494065 | 10.00733022 |
| 5184 phase | 12 | -1.120559076590 | 0.386489238760 | 10.00734581 |

## Reverse Timing Shares

Steady-state timing below excludes the first five phase rows per rank. The denominator is measured outer reverse time per rank-step:
`reverse_interaction + reverse_product0_transpose + reverse_product1_transpose + reverse_M0 + reverse_M1 + reverse_mpi_comm`.

| case | ranks | outer reverse s/rank-step | reverse interaction | reverse conv | layer 1 | layer 0 | reverse MPI comm | reverse M0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 770 | 1 | 0.088273 | 90.59% | 75.41% | 58.48% | 32.12% | 0.00% | 6.27% |
| 770 | 12 | 0.013568 | 67.68% | 42.37% | 42.93% | 24.75% | 21.93% | 6.76% |
| 5184 | 1 | 0.551557 | 89.86% | 77.32% | 57.06% | 32.80% | 0.00% | 7.35% |
| 5184 | 12 | 0.054637 | 82.12% | 65.85% | 52.50% | 29.62% | 8.16% | 6.94% |

Selected secondary stage shares:

| case | ranks | `linear_2^T` | `linear_1^T` | `linear_up^T` | gate+normalize | linear residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 770 | 1 | 4.26% | 3.46% | 1.92% | 1.08% | 0.94% |
| 770 | 12 | 5.47% | 5.69% | 5.41% | below 1% | 1.59% |
| 5184 | 1 | 4.45% | 3.31% | 1.01% | 1.17% | 0.94% |
| 5184 | 12 | 4.53% | 3.66% | 2.62% | 1.02% | 0.96% |

## Reverse Optimization Plan

- First target: `reverse_rrnlb_interaction_layer` reverse convolution. It is the dominant shared reverse cost for 770 and 5184, at 1 rank and 12 ranks.
- Prioritize layer 1 inside the reverse convolution. It consistently costs more than layer 0 and is the larger share of the measured reverse path.
- Compare the existing disabled sender-tiled reverse branch before writing a new algorithm. It is controlled by `SYMMETRIX_RRNLB_SENDER_TILED_REV`; segment size is controlled by `SYMMETRIX_RRNLB_SENDER_SEGMENT_SIZE`.
- Keep that comparison off by default. Treat it as an experiment because previous sender-owned reverse work regressed either drift or performance.
- If sender-tiled reverse is inaccurate or slower, stay within the current edge-parallel ownership model and reduce atomic/scratch pressure in the reverse-conv kernel instead of changing the algorithmic shape.
- Keep validation paired across 770 and 5184 atoms, 1 and 12 ranks. A tiny-case win that hurts 5184 should not be accepted as the default.

## Instrumentation Review

- The new timing hooks are gated behind `SYMMETRIX_RRNLB_PHASE_CSV`; production runs without the diagnostic env do not add the timing fences.
- The timing matrix confirmed the VTune interpretation and made the 12-rank split clearer: 5184 12-rank is still reverse-conv dominated, while 770 12-rank is both reverse-conv and reverse-MPI sensitive.
- Product transposes and `reverse_M0/M1` are measurable but secondary; oneMKL transpose work is not the first optimization target.
- Staff-engineer checkpoint: the next change should be an env-gated reverse-conv experiment with exact drift/force checks, not a default-path rewrite.

# Reverse Kernel Optimization Pass (2026-04-28)

## Goal

Improve the RRNLB pair/reverse kernel on Aurora, starting with `reverse_rrnlb_interaction_layer` reverse convolution, while preserving force/energy behavior across 770-atom and 5184-atom water workloads at 1 and 12 ranks.

## Plan

- [x] Review the current default edge-parallel reverse-conv kernel and the disabled sender-tiled branch.
- [x] Add a dedicated candidate PBS matrix that can force `SYMMETRIX_RRNLB_SENDER_TILED_REV` without the wrapper clearing it.
- [x] Run the sender-tiled candidate matrix for 770 and 5184 atoms at 1 and 12 ranks.
- [x] Compare speed, phase timing, and drift against the default reverse timing baseline.
- [x] If sender-tiled is both accurate and faster, tune segment size and choose a conservative default policy.
- [x] If sender-tiled is inaccurate or slower, implement a narrower edge-parallel reverse-conv optimization focused on reducing scratch/atomic overhead.
- [x] Rebuild after source changes and run short phase timing.
- [x] Validate the chosen candidate with 1000-step 770 and 5184 atom runs at 1 and 12 ranks.
- [x] Document the accepted optimization, rejected candidates, residual risk, and exact validation evidence.

## Design Notes

- Do not introduce rank-count special cases. Use workload/topology signals only if a default policy is needed.
- Keep experimental paths behind env gates until accuracy and performance are proven.
- Prefer preserving the existing edge-parallel ownership semantics unless the sender-tiled path proves clean.

## Candidate Results

- Sender-tiled reverse job: `8453386.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- Output root: `/home/skaiser/tests/water_benchmark_5184_symmetrix/runs/reverse_candidate_8453386.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- Result: rejected as default. It was accurate in short runs, but did not transfer to 770 12-rank performance. Best segment size `16` gave only small 1-rank/large-workload movement and regressed 770 12-rank from `1.550` to `1.523 ns/day`.
- Reverse vector-length isolation job: `8453396.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- Result: rejected. `SYMMETRIX_RRNLB_REVERSE_VECTOR_LENGTH=2` already produced massive drift and speed regression; larger vector lengths were also nonphysical, so the job was canceled after enough evidence.

## Accepted Change

- Added precomputed `conv_work_by_in_offsets` and `conv_work_by_in_indices` tables at model load time.
- Added a grouped-input edge-parallel reverse-conv path in `reverse_rrnlb_interaction_layer`.
- The new path keeps the existing edge-parallel edge ownership and force reduction, but removes per-edge scratch allocation, scratch zeroing, and scratch atomics for `h_up_adj`.
- It computes sender feature adjoints by input channel from precomputed work groups and emits one global accumulation per input channel.
- The path is default-on after validation. `SYMMETRIX_RRNLB_REVERSE_GROUPED_INPUT=0` disables it for fallback comparison.

## Short Phase A/B

- Job: `8453405.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- Output root: `/home/skaiser/tests/water_benchmark_5184_symmetrix/runs/reverse_grouped_8453405.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- Error scan found no `Segmentation fault`, `Abort`, `ERROR`, `NaN`, GPU page fault, or traceback entries.

| case | ranks | default ns/day | grouped ns/day | speedup | default reverse conv s/rank-step | grouped reverse conv s/rank-step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 770 | 1 | 0.323 | 0.367 | 13.6% | 0.066622 | 0.050799 |
| 770 | 12 | 1.546 | 1.640 | 6.1% | 0.005757 | 0.004380 |
| 5184 | 1 | 0.054 | 0.061 | 13.0% | 0.427251 | 0.329872 |
| 5184 | 12 | 0.501 | 0.558 | 11.4% | 0.035948 | 0.027411 |

## Production Validation

- Job: `8453411.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- Output root: `/home/skaiser/tests/water_benchmark_5184_symmetrix/runs/reverse_grouped_validation_8453411.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- All cases used 1000 steps without phase timing.
- Error scan found no `Segmentation fault`, `Abort`, `ERROR`, `NaN`, GPU page fault, or traceback entries.

| case | ranks | default ns/day | grouped ns/day | speedup | default pair avg s | grouped pair avg s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 770 | 1 | 0.319 | 0.362 | 13.5% | 134.89 | 118.96 |
| 770 | 12 | 1.571 | 1.668 | 6.2% | 25.167 | 23.657 |
| 5184 | 1 | 0.053 | 0.061 | 15.1% | 810.41 | 708.23 |
| 5184 | 12 | 0.475 | 0.528 | 11.2% | 88.284 | 79.320 |

Correctness remained matched within the existing benchmark noise:

| case | ranks | default drift | grouped drift | default delta K | grouped delta K |
| --- | ---: | ---: | ---: | ---: | ---: |
| 770 | 1 | 0.000284696452 | 0.000319737399 | 7.67008104 | 7.67080743 |
| 770 | 12 | 0.000307389658 | 0.000353469561 | 7.66962558 | 7.67025489 |
| 5184 | 1 | 0.001467819495 | 0.001460555813 | 2.00886016 | 2.00863186 |
| 5184 | 12 | 0.001469719055 | 0.001477019828 | 2.00878299 | 2.00860022 |

## Default-On Smoke

- Job: `8453430.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- Output root: `/home/skaiser/tests/water_benchmark_5184_symmetrix/runs/reverse_grouped_8453430.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.
- Purpose: confirm that the final rebuilt unset-env path now selects grouped-input behavior.
- Error scan found no `Segmentation fault`, `Abort`, `ERROR`, `NaN`, GPU page fault, or traceback entries.
- Default and explicit grouped phase timings matched after default-on:
  - 770 12-rank default/grouped reverse conv: `0.004373819` / `0.004375157 s/rank-step`.
  - 5184 12-rank default/grouped reverse conv: `0.027489072` / `0.027494129 s/rank-step`.

## Review

- The accepted change is not rank-specialized and does not change MPI/domain-decomposition semantics.
- It improves both small and larger water workloads and both 1-rank and 12-rank cases.
- It does not get the 770 one-node case to `2 ns/day`; the validated speed is `1.668 ns/day`, up from `1.571 ns/day` in this production A/B and from the earlier `~1.55 ns/day` baseline.
- The remaining gap is now less dominated by reverse convolution; next targets should be reverse MPI/halo overhead for 770 12-rank and any remaining layer-1 reverse-conv arithmetic/memory traffic.

# Follow-On Reverse/Pair Optimization Tranches (2026-04-28)

## Goal

Continue Aurora pair/reverse optimization after grouped-input reverse convolution until either no credible options remain or the 770-atom 12-rank benchmark improves by more than `0.2 ns/day` over the validated grouped baseline (`1.668 ns/day`). Preserve correctness and avoid regressing the 5184-atom workload.

## Plan

- [x] Re-rank remaining bottlenecks after grouped-input default-on validation.
- [x] Inspect reverse MPI communication and pack/unpack paths for avoidable payload or synchronization.
- [x] Inspect remaining layer-1 reverse-conv work for arithmetic or memory traffic reductions compatible with the grouped path.
- [x] Implement one narrowly scoped tranche at a time.
- [x] Rebuild after the first tranche.
- [x] Run a short full-node production timing after the first tranche.
- [x] Reject the comm internal-fence tranche: 300-step `8453447` produced 770 12-rank `1.662 ns/day` and 5184 12-rank `0.545 ns/day`, not a meaningful gain over the grouped baseline.
- [x] Implement a sender-segment grouped-input reverse variant for 12-rank atomics.
- [x] Test sender-segment grouped-input reverse across segment sizes on 12 ranks.
- [x] Reject the sender-segment grouped-input tranche: job `8453452` regressed 770 12-rank from default `1.626 ns/day` to `1.405`, `1.391`, `1.367`, and `1.311 ns/day` for segment sizes `4`, `8`, `16`, and `32`; 5184 12-rank similarly regressed from `0.554` to about `0.424-0.427 ns/day`.
- [x] Rebuild after rejecting sender-segment grouped-input so the benchmark binary again matches the accepted source.
- [x] Test grouped-input fused reverse: combine the grouped `h_up_adj` input pass with force/dE reductions to remove the duplicate work-table sweep while preserving edge-parallel semantics. It is controlled by `SYMMETRIX_RRNLB_REVERSE_GROUPED_FUSED` and defaults on for the candidate build.
  - Build completed successfully after the candidate change.
  - Phase A/B job `8453466` was canceled while still queued after code review found a device-side branch in the candidate kernel.
  - Candidate was revised to split fused and legacy grouped kernels at the host branch so the fused timing does not pay for legacy-path control flow.
  - Rebuild completed successfully and replacement phase A/B job `8453469` submitted.
  - Job `8453469` showed a small but real win: 770 12-rank `1.621 -> 1.666 ns/day`, 5184 12-rank `0.556 -> 0.575 ns/day`; reverse conv dropped `0.00437 -> 0.00400 s/rank-step` for 770 and `0.0270 -> 0.0247` for 5184. This is useful but below the `>0.2 ns/day` threshold.
- [x] Test oneMKL no-wait launch candidate: Kokkos uses in-order SYCL queues, so gate the internal oneMKL `wait_and_throw()` calls behind `SYMMETRIX_RRNLB_ONEMKL_WAIT`; after validation, no-wait is the default and `SYMMETRIX_RRNLB_ONEMKL_WAIT=1` restores the conservative wait behavior.
  - Build completed successfully and phase A/B job `8453487` submitted with fused reverse enabled on both sides.
  - Job `8453487` showed 770 12-rank `1.644 -> 1.700 ns/day` for wait vs no-wait, with matched drift; 5184 12-rank was effectively neutral (`0.574 -> 0.577 ns/day`).
- [x] Test existing grouped packed Kokkos linear backend against oneMKL no-wait for small 12-rank batches.
  - Phase A/B job `8453489` submitted.
  - Result: rejected. Packed backend regressed badly: 770 12-rank `1.690 -> 0.126 ns/day`; 5184 12-rank `0.575 -> 0.068 ns/day`.
- [x] Run longer no-profile production A/B for accepted grouped baseline behavior vs fused reverse plus oneMKL no-wait.
  - Production A/B job `8453492` submitted for 1000-step 770 and 5184 12-rank cases.
  - Result: keep the stack. Production 1000-step A/B improved 770 12-rank `1.668 -> 1.729 ns/day` and 5184 12-rank `0.528 -> 0.549 ns/day`; drift and delta-K matched baseline (`770 drift 0.000351 -> 0.000331`, `5184 drift 0.001456 -> 0.001467`).
  - This is below the `>0.2 ns/day` target, so continue only if another credible low-risk tranche remains; otherwise document residual bottlenecks.
- [x] Rebuild with fused reverse and oneMKL no-wait as the default path, preserving env overrides for rollback.
  - Rebuild completed successfully.
  - Default no-profile smoke job `8453498` submitted with all candidate env vars unset.
  - Default smoke result: 770 12-rank `1.711 ns/day`, 5184 12-rank `0.566 ns/day` for 300-step no-profile runs. Short-run drift matched the same 300-step noise pattern seen in prior timing-only runs.
- [x] Keep only changes that are accurate and improve 770 12-rank or 5184 12-rank timing.
- [x] Run 1000-step production validation for any candidate near the `>0.2 ns/day` acceptance bar.
- [x] Document rejected options and final residual bottlenecks.

## Follow-On Review

- Kept: fused grouped-input reverse conv and default no-wait oneMKL submissions on Kokkos' in-order SYCL queue. Rollbacks remain available with `SYMMETRIX_RRNLB_REVERSE_GROUPED_FUSED=0` and `SYMMETRIX_RRNLB_ONEMKL_WAIT=1`.
- Validated production gain over the grouped baseline is `+0.061 ns/day` for 770 12-rank and `+0.021 ns/day` for 5184 12-rank.
- The `>0.2 ns/day` target was not reached. The credible local pair/reverse tranches tested here are exhausted: comm internal-fence removal, sender-segment reverse accumulation, fused grouped reverse, oneMKL no-wait, and packed Kokkos linear backend.
- Remaining 770 12-rank bottlenecks are distributed across reverse conv, oneMKL transpose work, and reverse MPI/halo exchange. Getting another `>0.2 ns/day` likely requires a larger algorithmic communication change or broader forward-path work, not another small local reverse-kernel edit.

## Starting Point

- Branch: `perf/aurora-water-12r-transfer-2026-04-28`.
- Commit: `1634f2d` (`aurora: group reverse convolution input adjoints`).
- Validated grouped baseline: 770 12-rank `1.668 ns/day`, 5184 12-rank `0.528 ns/day`.
- Default-on grouped phase smoke: 770 12-rank reverse conv about `0.00437 s/rank-step`, reverse MPI about `0.0029 s/rank-step`.

# Current-Path Bottleneck And Redesign Pass (2026-04-28)

## Goal

Benchmark the newly committed fused-reverse/no-wait path, identify the current dominant bottleneck, and use that evidence to choose the next larger optimization path. If reverse remains dominant, redesign the reverse path; if forward now dominates, switch to a forward-path plan. Fused kernels are allowed only behind a measurement gate because launch reduction can lose to register pressure or reduced occupancy.

## Plan

- [x] Confirm the accepted gains are locked in a local commit and identify the exact commit: `857de95` (`aurora: fuse reverse input pass and relax onemkl waits`).
- [x] Submit a current-default 12-rank phase/profile benchmark for both 770-atom and 5184-atom water workloads: job `8454410`.
- [ ] Compare current phase timing against the previous grouped baseline and VTune hotspot mix.
- [ ] Decide whether the largest remaining cost is reverse, forward, oneMKL linear transpose, MPI reverse exchange, or a distributed mix.
- [x] If reverse remains dominant, design and implement the next reverse-path tranche with rollback env gates.
- [ ] If forward dominates, design and implement a forward-path tranche instead.
- [ ] Rebuild and run short smoke/phase validation for any source change.
- [ ] Run no-profile production A/B on 770 12-rank, and include 5184 12-rank when the change touches shared kernels.
- [ ] Compare drift, delta-K, and error scans before accepting any change.
- [ ] Commit only accepted source/harness changes and document rejected options.

## Initial Constraints

- Do not rely on rank-count special cases.
- Keep exact rollback knobs for any new path until it has production validation.
- Treat a fused kernel as a candidate, not a presumption; reject it if it increases register pressure or slows 12-rank.
- Preserve the committed fallback controls from `857de95`: `SYMMETRIX_RRNLB_REVERSE_GROUPED_FUSED=0` and `SYMMETRIX_RRNLB_ONEMKL_WAIT=1`.

## First Candidate

- The current reverse path still pays `linear_up^T -> x_up_adj` followed by a separate `x_up_adj -> node_feats_in_adj` accumulation kernel.
- Candidate: allow `rrnlb_apply_linear_transpose` to skip output zeroing and accumulate `linear_up^T` directly into `node_feats_in_adj`.
- Rollback: `SYMMETRIX_RRNLB_DIRECT_LINEAR_UP_ACCUM=0`.
- Expected upside is modest but clean: one fewer temporary clear/write/read and one fewer reverse-stage kernel after convolution.
- Build note: the first rebuild attempt failed before compilation because `cmake` was not in the shell `PATH`; retry with the explicit Aurora 25.190 / oneAPI 2025.2 module stack used by the benchmark harness.
- Rebuild result: module-loaded rebuild completed successfully.
- Current-default profile job `8454410` completed before the rebuilt binary timestamp, so it is a clean profile of commit `857de95`.
- Current 770 12-rank phase profile: speed `1.697 ns/day`; reverse interaction `0.007229 s/rank-step`, reverse conv `0.004034`, reverse MPI `0.002800`, `linear_up^T + input_adj_accum` `0.000790`.
- Current 5184 12-rank phase profile: speed `0.578 ns/day`; reverse interaction `0.033843 s/rank-step`, reverse conv `0.025140`, reverse MPI `0.004019`, `linear_up^T + input_adj_accum` `0.001556`.
- Direct-linear-up A/B job submitted: `8454447`.
- Prepared a second reverse-conv candidate in source, not yet rebuilt: model-load materialized grouped work arrays so the fused grouped reverse kernel can avoid indirect `work_by_in_indices -> work_*` gathers. Rollback: `SYMMETRIX_RRNLB_REVERSE_GROUPED_DIRECT_WORK=0`.
- Direct-linear-up A/B result from `8454447`: provisionally keep. 770 12-rank improved `1.689 -> 1.706 ns/day`; 5184 12-rank improved `0.576 -> 0.577 ns/day`. Drift and delta-K matched the short-run baseline. Reverse `linear_up^T + input_adj_accum` dropped from `0.000795` to `0.000692 s/rank-step` for 770 12-rank.
- Direct-work reverse-conv rebuild completed successfully.
- Direct-work A/B job submitted: `8454475`.
- Direct-work A/B result from `8454475`: keep for production validation. With direct-linear-up enabled on both sides, 770 12-rank improved `1.696 -> 1.727 ns/day`; 5184 12-rank improved `0.578 -> 0.589 ns/day`. Reverse conv dropped `0.004025 -> 0.003778 s/rank-step` for 770 and `0.025093 -> 0.023707` for 5184.
- Production validation submission first failed because `01:20:00` exceeded queue limits; script walltime reduced to `01:00:00`.
- Production validation job submitted: `8454494`.
- Production validation result from `8454494`: keep the combined stack, but it is not a `>0.2 ns/day` improvement.
  - 770 12-rank: `1.724 -> 1.745 ns/day`, `+0.021`.
  - 5184 12-rank: `0.552 -> 0.564 ns/day`, `+0.012`.
  - Drift/delta-K matched baseline noise: 770 drift `0.000317 -> 0.000301`, 5184 drift `0.001456 -> 0.001469`.

## Current Bottleneck Decision

- The newly committed fused/no-wait path is still reverse-heavy after profiling. In job `8454410`, 770 12-rank reverse interaction was `0.007229 s/rank-step`; reverse conv was `0.004034`, reverse MPI was `0.002800`, and `linear_up^T + input_adj_accum` was `0.000790`.
- Direct-linear-up removed the standalone input-adjoint accumulation kernel and reduced the linear-up tail, but the production win was small.
- Direct grouped-work arrays reduced reverse conv by avoiding the grouped-work indirection. This is the cleanest remaining local reverse-conv improvement, but production gain is still small.
- Remaining reverse cost is split between reverse conv and reverse MPI/halo exchange. The already-tested comm fence removal did not improve production timing, and changing the reverse communication algorithm would be a larger LAMMPS communication design rather than a contained SymmetriX kernel tranche.

## Review

- Accepted:
  - `SYMMETRIX_RRNLB_DIRECT_LINEAR_UP_ACCUM`, default on, rollback `=0`.
  - `SYMMETRIX_RRNLB_REVERSE_GROUPED_DIRECT_WORK`, default on, rollback `=0`.
- Rejected or exhausted in this cycle:
  - More local oneMKL wait/fence changes: already tested in the prior tranche.
  - Reverse sender segmentation: already regressed.
  - Reverse MPI internal-fence removal: already neutral/regressed.
  - Larger fused linear/reverse kernels: likely duplicate oneMKL work or increase register pressure; no clean design survived the phase data.
- The requested `>0.2 ns/day` 12-rank improvement was not achieved. The remaining path to that size likely needs a broader reverse communication redesign or a different algorithmic decomposition, not another local reverse-conv or transpose cleanup.

# Unified Reverse Communication Redesign (2026-04-28)

## Goal

Reduce the 12-rank 770-atom reverse communication tax without introducing a rank-count-specific or benchmark-specific code path. Preserve one default RRNLB reverse path that also remains correct and acceptable for larger 5184-atom workloads.

## Plan

- [x] Map the current Kokkos RRNLB reverse adjoint dataflow around `rrnlb_feat0_adj`, `comm->reverse_comm(this)`, `pack_reverse_comm_kokkos`, and `unpack_reverse_comm_kokkos`.
- [x] Confirm what portion of reverse communication is payload movement versus enforced ordering/fences and pack/unpack kernels.
- [x] Choose the simplest unified redesign that reduces communication ordering, payload, or post-communication gather work without changing the mathematical adjoint.
- [x] Implement the redesign with a rollback environment knob until validated.
- [x] Rebuild with the Aurora module stack used by the benchmark harness.
- [x] Run a short correctness smoke test against the rollback path.
- [x] Run 770-atom and 5184-atom 12-rank performance/phase validation.
- [x] Compare force/energy drift and delta-K against the rollback path before accepting.
- [x] Commit only if the change is numerically equivalent and performance-positive.

## Constraints

- Avoid splintering into separate 1-rank, 12-rank, or 770-only paths.
- Do not reduce model generality by assuming inactive channels or a fixed feature width.
- Treat pack/unpack kernel micro-optimization as secondary; existing VTune evidence says the larger issue is reverse communication ordering/latency.
- Keep rollback controls from earlier commits intact.

## Implementation Notes

- Added `SYMMETRIX_RRNLB_COMPACT_REVERSE_UNPACK`, default on, rollback `=0`.
- The new path initializes compact `feat0_adj_nodes` before reverse communication, builds an atom-index to compact-node map, and has RRNLB reverse unpack accumulate returned owner adjoints directly into compact node order.
- The atom-indexed `rrnlb_feat0_adj` buffer is still retained for ghost forwarding during LAMMPS reverse multi-swap communication, so the path is not rank-count-specific.
- Rebuild command completed successfully:
  - `cmake --build /home/skaiser/soft/lammps-aurora-water-perf/build_lean_25190 --target lmp -j 12`
- Phase A/B script added:
  - `tasks/run_aurora_compact_reverse_unpack_12r_matrix.pbs`
- Phase A/B job submitted:
  - `8454655`

## Phase A/B Result

- Job `8454655` completed cleanly.
- 770 12-rank: rollback `1.722 ns/day`, candidate `1.729 ns/day`; drift and delta-K matched.
- 5184 12-rank: rollback `0.588 ns/day`, candidate `0.587 ns/day`; drift and delta-K matched.
- Phase counters showed the reverse MPI stage was essentially unchanged:
  - 770 reverse MPI `0.002754653 -> 0.002752748 s/rank-step`.
  - 5184 reverse MPI `0.004284317 -> 0.004262935 s/rank-step`.
- Decision: this pair-side compact-unpack path is numerically valid but too small and slightly negative for 5184, so it is not the broader redesign we need. Move to the LAMMPS Kokkos reverse-comm ordering layer.

## LAMMPS Kokkos Reverse-Comm Ordering Plan

- [ ] Add an environment-gated Kokkos reverse pair comm variant that posts the receive before pack and removes the pre-receive device fence.
- [ ] Preserve the reverse swap order to avoid breaking multi-hop ghost reductions.
- [ ] Keep a rollback knob for direct A/B in one binary.
- [ ] Rebuild LAMMPS and run the same 770/5184 12-rank phase A/B.
- [ ] Accept only if the phase and production results show a meaningful positive change without drift.

## LAMMPS Ordering Implementation Notes

- Added `LAMMPS_KOKKOS_REVERSE_PAIR_POST_RECV_FIRST`, default on, rollback `=0`, in the local LAMMPS Kokkos pair reverse comm driver.
- The change keeps the reverse swap loop order intact, but posts `MPI_Irecv` before pair pack for each non-self swap.
- Rebuilt `/home/skaiser/soft/lammps-aurora-water-perf/build_lean_25190/lmp` successfully.
- Phase A/B script added:
  - `tasks/run_aurora_lammps_reverse_order_12r_matrix.pbs`
- Phase A/B job submitted:
  - `8454687`

## LAMMPS Ordering Phase A/B Result

- Job `8454687` completed cleanly.
- 770 12-rank: rollback `1.720 ns/day`, candidate `1.727 ns/day`.
- 5184 12-rank: rollback `0.584 ns/day`, candidate `0.585 ns/day`.
- Reverse MPI moved only slightly:
  - 770 reverse MPI `0.002747178 -> 0.002737218 s/rank-step`.
  - 5184 reverse MPI `0.004313149 -> 0.004281016 s/rank-step`.
- Decision: receive-before-pack alone is too small.

## LAMMPS Nonblocking Send Follow-Up

- Added `LAMMPS_KOKKOS_REVERSE_PAIR_NONBLOCKING_SEND`, default on, rollback `=0`, to replace blocking `MPI_Send` with `MPI_Isend` plus wait inside the same reverse swap.
- Updated `tasks/run_aurora_lammps_reverse_order_12r_matrix.pbs` so rollback disables both LAMMPS ordering flags and candidate enables both.
- Rebuilt `/home/skaiser/soft/lammps-aurora-water-perf/build_lean_25190/lmp` successfully.
- Combined ordering A/B job submitted:
  - `8454716`

## Final Lock-In Decision

- [x] Per user direction, accept the compact RRNLB reverse unpack path as a small positive movement and call the broader redesign out of scope for now.
- [x] Keep `SYMMETRIX_RRNLB_COMPACT_REVERSE_UNPACK=0` as the rollback knob.
- [x] Do not carry the LAMMPS Kokkos reverse-comm ordering/nonblocking-send source experiments in this SymmetriX commit.
- [x] Use job `8454655` as the acceptance benchmark: 770-atom 12-rank moved `1.722 -> 1.729 ns/day`; 5184-atom 12-rank stayed noise-level at `0.588 -> 0.587 ns/day`; drift and delta-K matched rollback.
- [x] Commit the accepted SymmetriX pair-side change and compact reverse A/B harness.
