# Safe holds and endpoint-only actuator identification

Research manuscript by **Shivam Gupta**:
**When Safe Holds Hide Actuator Gains: Exact Resolution-Constrained Observation Scheduling**.

This repository studies a known discrete state matrix, an unknown constant scalar
actuator gain, held inputs, endpoint-only state observations, and bounded process
disturbances. The paper derives the exact next-observation worst-case parameter
diameter, an exact safe duration frontier, a stable cancellation family, and an
optimal-resolution short-probe construction requiring at most one extra endpoint
under its stated feasibility assumptions.

The manuscript is prepared for L4DC 2027. It is **not an accepted publication**.
No acceptance probability or exhaustive novelty guarantee is claimed.

## Paper and evidence

- [Paper PDF](paper.pdf): 10 main-text pages, references and appendices; 4 figures.
- [LaTeX source](paper/manuscript.tex), [references](paper/references.bib).
- [Frozen experiment plan](EXPERIMENT_PLAN.md).
- [Literature and application assessment](RESEARCH_ASSESSMENT.md).
- [Independent test report](results/verification.txt).
- [Saved-data audit](results/saved_audit.json).

The main evaluation contains **2,400 episodes**, **576,000 physical transitions**,
and **145,296 endpoint observations** across six design rules, four simulated
plants and two disturbance regimes. A separate 200-episode sensitivity experiment
checks a fixed error in the known dynamics matrix. An analytic sweep has 2,880
parameter-duration cases. Every result is computed from executable code.

On the two oscillatory plants, longest holds use 10 observations without reaching
the gain-resolution target. A short minimax probe followed by long holds uses 11
and reaches it. The nonoscillatory and mixed-mode controls show that this failure
is not universal. A Gaussian proxy is a controlled objective comparator, not a
replication of a complete published controller.

## Reproduce

Tested with Python 3.9.6 and the package versions in `requirements.txt`.
A virtual environment is recommended:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 run_experiments.py
python3 run_mismatch.py
python3 analyze.py
python3 audit_saved.py
./build_paper.sh
```

`build_paper.sh` requires [Tectonic](https://tectonic-typesetting.github.io/).
It builds `paper/build/manuscript.pdf`. The supplied L4DC template is retained;
the year and running manuscript header are overridden, and the accepted-paper
option is disabled. The conference's linked template was labeled 2026 on
11 October 2026.

No model-service calls, credentials, downloaded datasets, or network requests
are needed to reproduce experiments. Tectonic may download TeX packages on the
first build. Timings and environment metadata vary by machine; the scientific
values and seeded trajectories are deterministic for the tested environment.

## Data schema

- `results/runs.csv`: one row per main episode in archive order, including seed,
  plant, gain, rule, noise regime, final diameter, observations, state peak,
  resolution time, input energy, consistency counts, and runtime.
- `results/physical_trajectories.npz`: `states[run,0:241,dimension]`,
  `inputs[run,0:240]`, `disturbances[run,0:240,dimension]`; two-dimensional
  systems use NaN padding in dimension 3. Run order matches `runs.csv`.
- `results/endpoints.csv.gz`: every endpoint, with zero-based `run_index`, time,
  posterior bounds, duration, command, norm, and diameter certificate.
- `results/summary.csv`: per-plant/regime/rule means and sample SDs.
- `results/contrasts.json`: paired RCD-minus-longest final-width differences,
  with 98.75% marginal bootstrap intervals allocated across four comparisons.
- `results/mismatch.csv`: sensitivity episodes, stopping on an empty posterior.
- `results/sweep.csv`: all deterministic sensitivity cases.

The adversarial noise depends on each policy and is not a common-noise paired
trial. Random-noise comparisons use the same physical-time tape on each seed.
An unresolved first target time is coded 241 (horizon 240); it is right censored.

## Scope

The implementation's closed forms assume scaled orthogonal dynamics and
Euclidean state/disturbance balls. The general support/gauge theory is broader.
Safety refers to every discrete physical step, not unmodeled continuous motion.
Exact state measurements, a constant gain, and valid noise/model-error bounds
are substantive assumptions. This is a theory and simulation contribution,
not a hardware, LLM, tracking, energy-savings or commercial-adoption benchmark.

Original code is MIT licensed. The manuscript is copyright Shivam Gupta.
The third-party conference class is excluded from the original-code license
and retains its source attribution.
