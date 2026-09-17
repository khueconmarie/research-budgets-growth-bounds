# Research Budgets and Bounds on Growth Effects

Replication materials for the Macroeconomic Dynamics submission manuscript, version 0.10, by Hyunkyu Lee (Department of Economics, Kyung Hee University).

The package reproduces the paper's analytical numerical examples, eight tables, and Figure 1 from ten public accounting observations and explicit model restrictions. It runs independently of the author's research workspace. A GPU, cloud account, proprietary data, and LaTeX are not needed for the numerical reproduction.

## Reproduce everything

Tested with Python 3.12.14. From the downloaded repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

On Windows, activate the environment with `.venv\Scripts\activate` before the last two commands. Python dependencies are pinned in `requirements.txt`. Internet access is needed to install them, but the reproduction itself uses only included files.

The command verifies input checksums, checks all ten accounting component sums, runs symbolic and numerical checks, rebuilds table fragments and Figure 1, and compares 16 regenerated result files with the frozen reference outputs. The three coverage-crossing rows also receive an independent 60-decimal calculation. A complete run took about 3 seconds on the author's macOS/Python 3.12 environment, excluding dependency installation. The final report is `checks/reproduction_report.json`; individual logs and numerical checks are in `checks/`.

The figure is written to `figures/accounting_bound_region.pdf` and `.png`. Publication table fragments and numerical result ledgers are written to `generated/`. Expected `.tex` output is checked byte-for-byte; numeric JSON values use relative tolerance `1e-11` and absolute tolerance `2e-13`. Plot file hashes are not compared because PDF metadata and graphics backends can vary across systems. Random verification grids use fixed seeds.

## Map from the paper to code

| Manuscript object | Builder / check | Main output |
| --- | --- | --- |
| Theorem 1 and resource-bound examples; Table 5 | `verify_drift.py` | `drift_results.json`, `drift_table.tex` |
| Feedback ceiling; Table 1 | `build_feedback_assets.py` | `feedback_results.json`, `feedback_cap_table.tex` |
| Observed accounting amounts; Tables 2 and 3 | `build_paper_assets.py` | `accounting_table.tex`, `bridge_table.tex`, `paper_assets.json` |
| Joint restriction and scalar root | `build_joint_assets.py` | `joint_results.json` |
| Minimum rule, crossing, and Table 4 | `build_information_assets.py` | `information_results.json`, `joint_bound_table.tex` |
| Coverage and productivity measurement; Tables 6-8 and Figure 1 | `build_measurement_assets.py` | `measurement_results.json`, three table fragments, figure |
| Two restricted optimality arguments | `verify_restricted_optimality.py` | `checks/restricted_optimality_verification.json` |
| Independent crossing calculation | `verify_precision.py` | `checks/high_precision_verification.json` |

The first builder also retains two auxiliary fragments (`headline_table.tex` and `drift_macros.tex`) used in the research workflow. These are not additional manuscript tables. The numerical kernel is in `analysis/`: `drift_core.py`, `joint_bound.py`, and `information_bounds.py`.

## Data and provenance

`data/accounting_observations.json` is the frozen source transcription. Its historical `purpose` field refers to the original rr03 extraction; the data are unchanged in v0.10. `data/accounting_observations.csv` is a flat convenience copy. `data/sources.json` provides official URLs, PDF SHA-256 hashes, and one-based PDF page locations. The original public prospectus and annual-report PDFs are linked rather than redistributed. `data/accounting_accounting_bounds.json` retains the original source-ledger calculations for comparison.

The primary sources are MiniMax's December 31, 2025 prospectus and Knowledge Atlas (Zhipu)'s December 30, 2025 prospectus, published by HKEX. The ten company-period observations overlap in time. They are not summed across periods or currencies, and interim observations are not annualized. Amounts are in thousands of the stated currency. Two printed MiniMax percentage entries differ from ratios of the reported raw amounts; the analysis consistently uses the raw-amount ratios. See `data/README.md` for details.

## Interpretation and scope

The code calculates sharp bounds for the specified necessary-condition resource block. Finite upper bounds apply to the included optimal regular-BGP subset. Candidate-set unboundedness is not an unrestricted global-optimality proof. The accounting observations are measured inputs; relative valuation, coverage, budget conversion, service growth, productivity drift, and the feedback ceiling are explicit theoretical restrictions. The scripts do not estimate these quantities or infer the world's AI growth effect.

The headline crossing, under the paper's stated reference restrictions, is `lambda = 0.6239603103388948`. At that exact crossing both standalone multiplier bounds equal approximately `1.291435`. Rounded `lambda = 0.624` is already above the crossing and gives `1.289161`; it is not an exact tie. At `lambda = 0.7`, the joint bound is approximately `0.324183`.

## Citation and assistance

Please cite Hyunkyu Lee, *Research Budgets and Bounds on Growth Effects*, manuscript version 0.10 (2026), and this repository version or commit. Machine-readable citation metadata are in `CITATION.cff`.

OpenAI Codex assisted with preparation and checking of code and with public-source inspection in September 2026. Numerical claims are accompanied by executable source and explicit checks. Responsibility for the research remains with the author. Contact: richardhk2@khu.ac.kr.
