"""Independent 60-decimal check of the coverage crossing and nearby table rows."""
from pathlib import Path
import json
import mpmath as mp

mp.mp.dps = 60
D = mp.mpf
alpha, beta, sigma = D(".04"), D(".96"), D(".85")
delta, k, budget, psi = D(".002"), D(".02"), D(".0192"), D("5")
share = D(14633) / D(84377)
A = 1 - alpha


def kap(g):
    return beta * mp.expm1(g) / (mp.expm1(g) + 1 - beta)


def eta(g):
    return beta * (mp.exp(g) - sigma) / (mp.exp(g) - beta * sigma)


def coefficient(m):
    if delta >= m * k:
        return D(0)
    return min(kap(m*k-delta)*((1-m)/m+eta(k)), kap(k-delta)*eta(k))


lo, hi = D(".5"), D(".6")
for _ in range(220):
    mid = (lo+hi)/2
    if mid*coefficient(mid/psi) < budget:
        lo = mid
    else:
        hi = mid
w = (lo+hi)/2
threshold = w/(psi*share)
feedback_bound = w/(A-w)
rows = []
for label, coverage in (("exact_crossing", threshold), ("rounded_crossing", D(".624")), ("above_crossing", D(".7"))):
    c = coefficient(coverage*share)
    wa = min(A, budget/c) if c else A
    wj = min(wa, w)
    rows.append({"case": label, "coverage": str(coverage),
                 "multiplier_composition": str(wa/(A-wa)),
                 "multiplier_feedback": str(feedback_bound),
                 "multiplier_joint": str(wj/(A-wj))})
assert abs(threshold-D(".623960310338894788743")) < D("1e-20")
assert abs(feedback_bound-D("1.291435130917806")) < D("1e-14")
reference = json.loads((Path(__file__).parent/"generated/information_results.json").read_text())
for a, b in zip(rows, reference["crossing_rows"]):
    assert a["case"] == b["crossing_label"]
    for name in ("coverage", "multiplier_composition", "multiplier_feedback", "multiplier_joint"):
        assert abs(D(a[name])-D(str(b[name]))) < D("2e-12"), name
report = {"precision_decimal_digits": mp.mp.dps,
          "coverage_threshold": str(threshold), "rows": rows,
          "comparison_to_main_code_tolerance": "2e-12"}
(Path(__file__).parent/"checks/high_precision_verification.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
