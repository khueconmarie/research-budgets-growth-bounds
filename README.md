# Research Budgets and Bounds on Growth Effects

**Replication release 0.11 — September 17, 2026.** Author: Hyunkyu Lee, Kyung Hee University.
This repository accompanies the main manuscript and its separate online supplement,
prepared for *Macroeconomic Dynamics*. It contains public accounting observations,
source provenance, analytical routines and deterministic table/figure builders.

The paper asks when a small research budget excludes a large long-run response to
cheaper research computing. The central result is the sharp lower resource
coefficient, its failure under productivity uncertainty, and restoration through
composition, feedback or research-labor growth restrictions. The minimum of two
standalone bounds is a corollary, not the principal claim of novelty.

## Download and run

Choose **Code → Download ZIP** on GitHub, or clone the repository. A commit-specific
ZIP freezes the version used by a manuscript; that full commit is recorded in the
manuscript's data/code statement and in the local submission instructions.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

Test environment: Python 3.12.14, NumPy 2.3.5, Matplotlib 3.11.1, SymPy 1.14.0,
mpmath 1.3.0. Dependencies are pinned. The complete numerical run takes approximately
3–5 seconds on the author's local Apple computer (environment-dependent). It requires
no GPU, credentials, original workspace, LaTeX or network after dependency installation.
The raw data and expected values are included; no live extraction occurs during a run.

The runner verifies four input hashes and all ten accounting component sums, executes
the stages in dependency order, compares every expected TeX output byte for byte and
JSON numbers at relative tolerance 1e-11 / absolute tolerance 2e-13, and writes
`checks/reproduction_report.json`. Expected outputs are reference values, not a
substitute for rerunning the calculations. PNG/PDF figure existence is checked;
PDF binaries are not compared byte for byte because metadata can vary.

## Output map

| Manuscript object | Generator | Output in `generated/` or `figures/` |
|---|---|---|
| Theorem 1 and candidate checks | `verify_drift.py` | `drift_results.json`, `drift_macros.tex` |
| Table 1: feedback at two decline allowances | `build_revision_assets.py` | `feedback_comparison_table.tex` |
| Table 2: ten disclosed company-periods | `build_paper_assets.py` | `accounting_table.tex` |
| Table 3: parameter status | manuscript text | Selected restrictions, no estimation |
| Table 4: separate and joint bounds | `build_information_assets.py`, then `build_revision_assets.py` | `joint_bound_table.tex` |
| Table 5: three information restrictions | `build_revision_assets.py` | `information_comparison_table.tex` |
| Table 6: acquisition vs user-cost budgets | `build_revision_assets.py` | `budget_definitions_table.tex` |
| Figure 1: binding information and feedback sensitivity | `build_revision_assets.py` | `information_frontier.pdf`, `.png` |
| Supplement Table S1 | `verify_drift.py` | `drift_table.tex` |
| Supplement Table S2 | `build_paper_assets.py` | `bridge_table.tex` |
| Supplement Tables S3, S5, S6 and Figure S1 | `build_measurement_assets.py` | `coverage_table.tex`, `normalization_table.tex`, `proxy_witness_table.tex`, `accounting_bound_region.pdf`, `.png` |
| Supplement Table S4 | `build_revision_assets.py` | `primitive_sensitivity_table.tex` |
| Restricted optimality algebra | `verify_restricted_optimality.py` | Check log/JSON |
| Exact transfer crossing at 60 decimal digits | `verify_precision.py` | `checks/high_precision_verification.json` |

The former feedback and headline fragments are retained as auxiliary numerical
checks; the active paper uses the output map above. `reproduce.py` runs the early
builders before the final v0.11 builder that writes the current Table 4.

## Observations and maintained restrictions

`data/README.md` and `data/sources.json` record original public filing URLs, PDF hashes,
page locations, currencies, periods, classification boundaries and two discrepancies
in printed MiniMax percentages. Amounts are not pooled across currencies or
overlapping periods. The original raw observation ledger is unchanged; its older
extraction-version field documents provenance, not the replication release version.

The observed minimum is 14633/84377. Relative valuation `r`, the transferred target
activity cost share `lambda`, future persistence, absolute social-resource factor
`Lambda_E`, service-growth floor `k`, productivity-decline allowance `delta` and
feedback ceiling `Psi` are maintained restrictions. Lambda is not sample completeness
or a measured aggregate share. The 1.92% world R&D ratio supplies a scale reference;
using it as a planner-budget ceiling is an additional allocation/valuation restriction.

The illustrative exact crossing .6239603103 is conditional on the declared inputs.
At delta=.002, feedback-only bounds for Psi=1,2,5,10 are .173617, .335939, 1.291435
and unbounded. A separate matched labor-growth floor of .002 gives .0909602565
without composition or feedback information. Equal numerical acquisition and
user-cost budget ceilings describe different restrictions, not one account twice.

## Scope of verification

Finite resource bounds also hold for the globally optimal regular-BGP subset.
Sharpness is established for the necessary-condition resource block. Candidate
feasibility, first-order conditions, TVCs, and restricted one-control-block global
comparisons do not certify simultaneous-control global optimality. No regression,
GPU experiment, transition solver or world-AI growth estimate is performed here.
The new user-cost coefficient and labor restriction are independently checked on
primitive grids; analytical proofs, rather than grids, establish the results.

`publication_manifest.json` records SHA-256 for every published content file except
itself. `.gitattributes` fixes LF line endings. See `CITATION.cff` for citation metadata.
