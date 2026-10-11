# Decision-certified control: evaluation plan
Recorded on 11 October 2026 before the new study's benchmark runs.

Question: can two-sided certificates compress a full unknown A,B model history
while certifying the safety-filter action and its objective gap to the full-data
robust controller? This replaces the scalar-gain investigation.

Theory uses row-wise polyhedral set membership with box process disturbances,
a compact prior containing the true matrix, exact state measurements, and a
continuous-input Euclidean projection safety filter. No stability or parameter
consistency is inferred from the one-step cost certificate. A common prior-safe
linear backup provides recursive feasibility in the chosen benchmarks.

Main plants: coupled thermal network (4 states, 2 inputs), discretized mechanical
chain (4,2), nonnormal network (6,2), and dense stable network (8,3). Every entry
of A and B is unknown within an explicitly supplied box prior. Main training
histories have 1,024 physical transitions. A design pilot precedes the main runs. Twelve independent seeds per plant
and two bounded disturbance regimes: uniform box and signed boundary noise.
Control queries use 96 steps per episode and aggressive bounded reference inputs.
Retain all episodes and every transition. Query benchmarks compare the same
state/reference against full data; closed-loop trajectories are reported
separately because policies visit different states.

Methods: full-data robust filter; decision-certified compression at objective
gap tolerances 1e-5 and 1e-3; fixed-budget random data; fixed-budget greedy
log-determinant regressor selection; rolling-window data; prior-only filter.
The fixed budget is 2*n*(n+m), capped by history length. These are explicit
baselines, not claimed replications of Li et al.'s centroid-volume algorithm.

Primary metrics: objective gap against full-data optimum, certificate validity,
action error versus its strong-convexity bound, retained scalar constraints,
retained raw-transition IDs, archive LP calls, local/archive/total wall time,
state/input safety. Report all failures, not just successful certificates.
Secondary: filtered-input intervention, reference tracking proxy and actual
closed-loop state norms. No universal wall-clock speedup is assumed.

Ablations: remove optimistic witnesses (safe but no useful regret certificate),
remove selected data (optimistic action may be unsafe), disable reuse, and
vary approximation tolerance. Scaling: history length 128,512,2048,8192 and
state dimension 2,4,8,12 with two inputs. Separate append-only streaming tests
screen model witnesses against every incoming constraint and audit validity
before accepting a reused lower bound. Noise-bound mismatch is an assumption
stress test, not part of the valid-model theorem.

Independent verification: vertex-enumerated two-dimensional support problems,
explicit controller action enumeration for one input, dual feasibility and
gaps for every extracted support certificate, compaction preserving both
sides of the objective sandwich, nested-data witness invalidation, and a
regular-polygon construction showing the need for growing data to preserve
all model support directions. Scientific comparisons exclude timing fields.

Pilot amendment, before main data collection: unconstrained interior queries
rarely activate the filter. The query benchmark therefore declares 75%
nominal-row-aligned boundary stress and 25% random interior queries. The
nonnormal plant has open-loop unstable dynamics and a prior-certified linear
backup; the other plants retain zero-input backups. Closed-loop results are
kept separate and are not selected for constraint activation.

Secondary closed-loop evaluation (specified before its execution): three seeds
3000--3002 per plant and regime, 96 physical steps per policy. Reference commands
hold a signed full-amplitude request for 16 steps, with a fixed per-seed random
phase pattern shared by policies. Actual disturbance tapes are shared by physical
time. A failed solve stops that policy and is retained as a failure. These runs
are separate from the 12-seed common-query primary benchmark.
