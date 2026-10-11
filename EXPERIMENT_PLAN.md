# Frozen evaluation protocol
Written before the main simulation runs on 11 October 2026.

Question: for known scaled-orthogonal state dynamics and an unknown constant
scalar actuator gain, how does the hold duration affect exact endpoint
set-membership resolution while keeping every discrete intermediate state safe?

All methods share the same exact robust safe-input intervals and the same
endpoint update. Methods: longest feasible hold; fixed hold at most six steps;
one-step probing; maximum worst-case resolution; Gaussian-information proxy;
resolution-constrained longest hold, target diameter 0.08. Each picks maximum
safe input magnitude, with a deterministic tie toward positive input. These
are controlled design-rule comparisons, not replications of full published
controllers. No parameters are tuned on the main simulation results.

Four declared plants: rotation period 8, damping .98; rotation period 24,
damping .995; nonoscillatory isotropic contraction .94; 3D rotation period 12
plus a fixed orthogonal axis, damping .98, normalized input direction (1,0,.3).
Disturbance radius .001; safe Euclidean ball radius 1; maximum input .25;
maximum duration 24. Prior [.5,1.5]. Actual gain Uniform[.6,1.4]. Initial
state uniform in the ball of radius .2. Physical horizon 240. 50 seeds
(1000..1049) per plant. Random unit-ball disturbances use the same physical
time-indexed tape for every method on a seed. A second adversarial experiment
uses policy-dependent, admissible disturbances directed to keep the endpoint
as close as possible to the current prior-midpoint response. It tests a
worst-case mechanism, not expected performance or a common-noise paired trial.

Retain every method, plant, seed, and noise regime. Primary outcomes: final
parameter interval diameter; number of endpoint observations; fraction of
physical time with an interval diameter <= .08; maximum state norm; prior
containment failures; failed batch certificates. Include target reach time
(right censored at horizon), input energy, and wall-clock cost as secondary
metrics. Report means and sample standard deviations and paired bootstrap
intervals for the declared RCD-minus-longest comparisons under random noise.
No claim of universal dominance or asymptotic consistency under bounded noise.

Mechanism study: frequency/damping/duration sweep; exact periodic cancellation;
damping limit formula; comparison of certificates to independent conic
optimization on random instances; independent endpoint inversion; adversarial
witnesses attaining the claimed one-observation worst-case diameter.
