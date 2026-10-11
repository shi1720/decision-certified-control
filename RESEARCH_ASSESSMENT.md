# Research positioning and evidence

Reviewed 11 October 2026. This document records the scope of the contribution, not a certification that no related idea exists or a forecast of acceptance.

The current paper replaces the earlier scalar-gain study with unknown full A and B matrices, a coupled upper/lower control-cost certificate, post-query compression on both sides, finite polyhedral refinement, append-only witness screening, and multivariable empirical validation. All results describe this new study.

## Closest work

- Jingyuan Li, Dawei Shi, Ling Shi, *Online Coreset Selection for Learning Dynamic Systems*, arXiv:2506.22804v2 (2026). Primary: https://arxiv.org/abs/2506.22804 . The full manuscript was read for its row polytope, centroid/geometric selection, redundancy removal, volume convergence, and noise-bound discussion. Dynamic-system coresets and row-polyhedral identification are prior art. DCC selects a query-specific safe-action/cost certificate; it does not claim volume convergence, identification consistency, or a fixed-size uniform approximation. The published geometric algorithm is not implemented as a baseline; that is an explicit empirical limitation.
- Yingying Li et al., *Learning the Uncertainty Sets of Linear Control Systems via Set Membership: A Non-asymptotic Analysis*, ICML 2024, PMLR 235:29234–29265. Primary: https://proceedings.mlr.press/v235/li24ci.html . The uncertainty-set convergence analysis addresses identification accuracy rather than the local decision sandwich. DCC requires containment, not an identification rate.
- Notarstefano and Bullo, *Distributed Abstract Optimization via Constraints Consensus: Theory and Applications*, IEEE TAC 56(10):2247–2261, 2011, https://doi.org/10.1109/TAC.2011.2164020 . Bounded-size optimization bases and finite exchange are established. The author manuscript was inspected. Sparse exact bases alone are not a new contribution.
- Carlone et al., *Distributed Random Convex Programming via Constraints Consensus*, arXiv:1207.6226, subsequently SIAM JCO, https://doi.org/10.1137/120885796 . Active/vertex constraint exchange and control applications are close foundations. Our two summaries encode opposite directions of model-set containment, not a new distributed-consensus algorithm.
- Calafiore and Campi, *The Scenario Approach to Robust Control Design*, IEEE TAC 51(5):742–753, 2006, https://doi.org/10.1109/TAC.2006.875041 . Support constraints are classical; DCC's retained data and witnesses are deterministic containments, not scenario generalization bounds.
- Boyd and Vandenberghe, *Convex Optimization*, 2004, https://web.stanford.edu/~boyd/cvxbook/ . LP duality, KKT optimality, and sparse conic representations are standard ingredients, explicitly attributed.
- Wabersich and Zeilinger, *A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems*, Automatica 129:109597, 2021, https://doi.org/10.1016/j.automatica.2021.109597 . Safety filtering is prior art. DCC compresses its model representation and certifies one-step intervention cost under an already supplied invariant set.
- Berberich et al., *Data-Driven Model Predictive Control With Stability and Robustness Guarantees*, IEEE TAC 66(4):1702–1717, 2021, https://doi.org/10.1109/TAC.2020.3000182 . Their finite-horizon stability/robustness results are not obtained by this one-step filter.
- Rickard, Abate, Margellos, *Data-Driven Certificate Synthesis*, Automatica 185:112798, 2026, https://doi.org/10.1016/j.automatica.2025.112798 . The publisher text was inspected. It addresses certificate synthesis with statistical compression guarantees, not an archive-relative safe-action intervention sandwich.

Additional searches combined decision compression, safety-filter coresets, two-sided uncertainty representations, set-membership control, and sparse support certificates. Searches cannot prove absence of prior art. The defensible claim is the specified synthesis of the two complementary summaries, their joint compaction and streaming validity, with executable evaluation. The elementary ingredients are not presented as discoveries.

## Evidence and limitations

The artifact retains every primary training history, query and action, all physical closed-loop trajectories, failure lists, and portable streaming examples. Eight independent check groups pass. The audit checks containment and sequential consistency, recomputes intervention objectives, and verifies numerical sandwich/action/size bounds. Computational verification is to declared floating-point tolerances; exact-arithmetic proofs do not certify arbitrary real software deployments.

Uniform and boundary noise are reported separately. Boundary-noise fixed-budget subsets match full-data results, and closed-loop intervention improvements are small. DCC is not universally faster than subset heuristics. Disabling compaction improves reuse but grows retained information, a relevant tradeoff rather than a hidden negative result.

## Application hypothesis

The archive/client split may be useful for supervisory building-control filters or industrial edge systems with known safe backups. An archive can retain fleet/device identification histories while edge clients retain decision certificates. This is a possible product architecture, not a market-size, savings, deployment, or patentability result. No claimed market numbers enter the paper. A commercial validation would need measured state/error models, archive availability and network timing, worst-case scheduling, hardware constraints, and application-specific demand. Patent novelty and freedom to operate would require a separate professional search; this literature review does not establish either.

## Submission fit

The mathematical topic fits learning, uncertainty estimation, optimization, and robust data-driven control. The main manuscript uses the conference-linked PMLR class with the year/header updated to 2027, ten main pages, and references/appendices following. No accepted volume or editors are fabricated. The only named author is Shivam Gupta. No unverified claim about the author's specific contribution is added.
