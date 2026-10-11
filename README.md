# Decision-Certified Compression for Safe Control with Unknown Linear Dynamics

Research artifact by **Shivam Gupta**, prepared for L4DC 2027. This is a full-matrix study of a deterministic archive/controller certificate architecture, with executable experiments and retained numerical evidence. It does not claim acceptance or hardware validation.

Selected data enlarge the uncertainty set and certify a safe action. Feasible full-history model witnesses relax the robust safety filter and supply a lower intervention-cost bound. Their gap decides whether the controller can accept locally or must query the archive. Sparse LP and QP supports compact both sides after acceptance. Incoming data must screen cached model witnesses.

The exact-arithmetic bounds are at most `2*n*(n+m)` selected scalar data inequalities and `m` row-model witnesses for box constraints and row-separable full-matrix uncertainty. Prior constraints are always retained. This bounds post-query local information, not archive memory, temporary refinement memory, iteration count, or total computation. The supplied implementation validates floating-point witnesses to stated tolerances; it is not an interval-certified control library.

## Contents

- `paper/main.pdf`: complete submission manuscript; ten main pages followed by references/appendices.
- `src/certified.py`: full-history support oracle, robust filter, two-sided cache, append-only screening.
- `src/portable.py`: standalone controller using exported selected inequalities and witnesses, without raw history.
- `benchmarks.py`: all four plants, prior-safe sequential data, common-query generator, subset baselines.
- `run_study.py`, `run_all.py`: primary experiment, with every method and failure retained.
- `run_closed_loop.py`, `run_streaming.py`, `run_ablation.py`, `run_scaling.py`: secondary studies.
- `results/`: all primary transition archives, query/action arrays, closed-loop trajectories, streaming payloads, scalar counterexample, CSV summaries, timing records, and audit.
- `tests/`: independent vertex enumeration, analytic one-input projections, compaction/containment, append-ID and portable-controller checks.
- `EXPERIMENT_PLAN.md`: design specification and pilot amendments recorded before the corresponding runs.
- `RESEARCH_ASSESSMENT.md`: related-work positioning and limits of novelty/commercial claims.

## Reproduce

Python 3.9 or newer; the recorded environment used Python 3.9.6, NumPy 1.26.4, SciPy 1.13.1, pandas 2.2.2, and matplotlib 3.9.4 on arm64 macOS. Install the versions in `requirements.txt` in a virtual environment. Run from this repository's root:

```sh
python3 -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_all.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_closed_loop.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_streaming.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_ablation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_scaling.py
python3 analyze.py
python3 build_paper.py
```

The experiment scripts overwrite their own results. Preserve the supplied release if comparing reruns. `run_all.py` uses at most two processes. Secondary scripts can be run sequentially. Full execution took minutes per experiment on the recorded machine; different environments and network-free solver setups change timing. `analyze.py` independently audits all retained data and produces manuscript figures and summaries. `build_paper.py` synchronizes the generated full table and calls the installed `tectonic` compiler using the conference-linked class file. Tectonic may download standard TeX packages on its first run.

To verify the delivered evidence without rerunning benchmarks:

```sh
python3 -m unittest discover -s tests -v
python3 analyze.py
```

The primary study comprises 96 histories, 98,304 sequential training transitions, 9,216 common queries, and 64,512 method-query evaluations. There are 168 closed-loop episodes (16,128 physical steps), 720 streaming queries, 768 scaling queries, and 768 ablation method-query evaluations. Both disturbance regimes share a parameter draw within a seed; bootstrap intervals cluster both regimes and all queries by the 12 independent seeds per plant.

## Reading the results

`results/audit.json` records checks and aggregate counts. `results/summary.csv` records per-plant/method means, seed-cluster bootstrap intervals, retention, archive calls, and timing splits. `results/noise_envelope_stress.csv` holds a worst-case exposure audit under larger deployment noise bounds, not a Monte Carlo violation rate. The main table reports scalar inequalities; fixed-budget baselines retain whole transitions and have a larger inequality budget. Boundary-noise subsets match full data numerically, an explicitly reported negative control. DCC is generally slower than the small-subset heuristics and faster than the full-data implementation in these primary experiments. No network latency was simulated.

The theoretical safety statement assumes exact measured states, a containing compact prior, correct known disturbance bounds, fixed dynamics, and a prior-safe invariant-set backup. One-step command intervention does not imply closed-loop asymptotic stability or a long-horizon regret guarantee. See the manuscript's limitations and numerical-verification appendix.

## Version history

This repository supersedes the earlier scalar held-input identification study. The original version is preserved under Git tag `v1-scalar-study`; it is not the present submission and is not pooled with this study's experiments. The repository was renamed from `safe-dwell-identification` to reflect the new contribution.

Code is MIT licensed. The manuscript and accompanying generated figures/data are released under CC BY 4.0; see `LICENSE` and `DATA_LICENSE.md`.
