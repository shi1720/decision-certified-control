# Research positioning and application assessment

Search date: 11 October 2026. This is an assessment of the completed study,
not a claim that keyword searching establishes absence of all prior art.

## Research question and contribution boundary

Can a safe held command guarantee useful actuator-gain resolution when only
its endpoint is observed? The paper supplies an exact conditional frontier,
a strictly stable cancellation family, and a one-extra-observation sufficient
construction under an explicit safe full-amplitude probe assumption.

The mathematics uses established set-membership geometry, support functions,
Minkowski sums, and geometric-series identities. Novelty must be assessed on
the particular synthesis, sharp held-input characterization, stable separation,
and reproducible controlled study. It would be inaccurate to claim invention
of safe self-triggered learning, sampling design, or bounded-error estimation.

## Closest literature and differences

| Work | Existing contribution | Difference retained in this paper |
| --- | --- | --- |
| [Zhang, Liu & Shi, 2019](https://arxiv.org/abs/1912.06978) | Aperiodic set-membership estimation, uncertainty-aware self-triggered min-max MPC, feasibility and stability | Exact endpoint resolution of held scalar-gain probes and explicit cancellation severity, rather than a new MPC framework |
| [Dhiman et al., 2023](https://doi.org/10.1109/TAC.2021.3137059) | Bayesian model learning, probabilistic safe self-triggered control | Deterministic bounded-disturbance resolution and explicit observation-gap design |
| [Cohen, Belta & Tron, 2022](https://arxiv.org/abs/2208.05955) | Robust barrier functions and integral set-membership reduction of actuator uncertainty | Endpoint-only dwell selection and accumulated gain sensitivity |
| [Tanaskovic, Fagiano & Morari, 2014](https://doi.org/10.1016/j.automatica.2014.10.058) | Optimal worst-case information radii, general regressor dynamics and convex input constraints; maximal pulses already optimal in a special case | Integer endpoint timing, accumulated process-disturbance geometry and robust unobserved state-prefix constraints; scalar minimax geometry and maximal probing are not new principles |
| [Casini, Garulli & Vicino, 2011](https://doi.org/10.1109/TAC.2011.2107091) and [2012](https://doi.org/10.1016/j.automatica.2012.08.016) | Optimal worst-case scalar-gain input design with binary/quantized sensors | Exact endpoint-state measurement with dynamic noise accumulation and prefix safety; no claim to invent scalar-gain experiment design |
| [Pronzato & Walter, 1990](https://doi.org/10.1016/0378-4754(90)90013-9) | Bounded-error experiment design by compatible-set size | Exact state-constrained held-input specialization; no claim to invent the design objective |
| [Marvel & Williams, 2012](https://doi.org/10.1186/1752-0509-6-21) | Bounded-error design of measurement times | Robust unobserved-prefix constraints and scalar closed forms |
| [Walz et al., 2018](https://doi.org/10.1016/j.compchemeng.2017.12.016) | Global constrained min-max bounded-error design | Finite scalar frontier exploiting held-gain structure |
| [Gonzalez et al., 2024](https://arxiv.org/abs/2410.19629) | Slow-sampling statistical identification and aliasing | Known discrete A and cancellation in an unknown-gain regressor, not continuous-time frequency reconstruction |
| [Dozzi et al., 2026](https://arxiv.org/abs/2605.13120) | D-optimized nonuniform sampling for multisine identification | Safe bounded-error endpoint probing and worst-case resolution |

Full texts were inspected for Zhang, Dhiman, Cohen, Gonzalez, Dozzi, Kosut,
Marvel, Tanaskovic, and both Casini papers. The Tanaskovic comparison was added
during the final expanded review and the contribution wording narrowed accordingly. Official publisher descriptions and metadata were inspected for
Pronzato and Walz. Queries included self-triggered identification, safe input
design, bounded-error sampling, endpoint identification, gain uncertainty,
and sampling-dependent excitation. This is a targeted review, not a systematic
review with exhaustive database coverage. The 1987 paper "Optimal sampling
schedule for parameter estimation of linear models with unknown but bounded
measurement errors" (DOI 10.1109/TAC.1987.1104535) was also located; its full
text was not obtained, so no detailed comparative claim is made about it.
That unresolved historical comparison is a residual novelty risk.

## Avoiding overlap with the author's existing manuscripts

The supplied research collection concerns stochastic tool-cache coupling,
retry reliability, agent-choice auditing, personal memory, prefix-cache
retention, financial exposure routing, deposit control, undo/compensation,
payment and capability contracts, noisy bandits, private participation, and
agent tool exposure. Their indexed titles and manuscript abstracts were
reviewed. The current study concerns discrete physical dynamics, robust prefix
safety, and observation timing for gain identification. No previous manuscript,
result table, dataset or figure was reused. General themes of uncertainty and
identification are shared, but the model, question, derivations and experiments
are distinct.

## Application and market evidence

[NIST's Wireless Systems for Industrial Environments program](https://www.nist.gov/programs-projects/wireless-systems-industrial-environments)
identifies joint time, reliability, energy and scale constraints and the need
for measurement-based selection and deployment of industrial wireless systems.
[NIST's industrial channel-characterization study](https://www.nist.gov/publications/channel-characterization-and-delay-optimization-industrial-wireless-control-systems)
quantifies reliability/latency challenges in industrial wireless environments.
The [US Department of Energy's sensors and controls portfolio](https://www.energy.gov/cmei/buildings/about-sensors-and-controls)
includes wireless sensor networks, analytics and adaptive controls.

These primary sources support the operational setting, not a market-size or
customer-demand estimate. A plausible product is a commissioning/certification
library attached to a telemetry scheduler. Inputs are an existing dynamics
model, actuator-gain range, bounded residual model, safe set and duration cap;
outputs are feasible commands and a guaranteed next-observation resolution.
The simulation suggests benchmarking a one-probe-then-hold implementation
before developing a more complex optimization product.

Potential users include industrial automation integrators and operators of
networked sensing/actuation fleets. The study does not establish purchase
intent, pricing, startup viability, physical energy savings, or a patentable
claim. There was no customer study and no comprehensive patent search. Public
release is authorized by the user's request to share the code; no patent
filing is represented as completed or recommended by this assessment.

## L4DC fit and acceptance risk

The work fits system identification, data-driven control, optimization and
robustness. Its strengths are explicit assumptions, sharp bounds, a stable
separation, an optimal-resolution remedy, full retained experiments and
independent numerical checks. Its main risks are the restricted scalar-gain
model, elementary constituent geometry, historical prior-art overlap,
absence of hardware, and the fact that simple short probing performs well.
A high acceptance probability cannot be supported. This is a complete research
manuscript suitable for critical consideration, not evidence of acceptance.

A useful next independent scientific review would test whether the sharp
held-input result is already implicit in older bounded-error input/sampling
design literature and whether the theoretical specialization is substantial
enough for L4DC. No independent human or external peer review has been claimed.
