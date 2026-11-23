"""Score the analysed scenes: the paper's metrics and tests, plus correctness.

Joins results/*.jsonl (one file per system, from analyse.py) with the ground truth in
scenes.json. This is the only experiment step besides spawning that reads scenes.json.

  python3 -m agentic_uav.experiment.score            (agent container; needs scipy)
Writes results/summary.md and results/summary.json and prints the tables.

The paper's metrics, per system
  processing time per sample; detection confidence (YOLO's top score, or the LLM's
  self-reported one); persons detected (samples with person_detected); ARR, Action
  Recommendation Rate (non-empty recommended_actions); CAR, Contextual Analysis Rate
  (surrounding_features longer than 50 characters).
Its tests: Shapiro-Wilk and Levene, then one-way ANOVA + Tukey HSD (3+ systems) or
Welch's t (2), Mann-Whitney U where normality fails, Cohen's d and rank-biserial r,
chi-square (Fisher's exact for a 2x2 with small counts) for rates; alpha 0.05. Like the
paper it treats systems as independent samples, although they saw the same scenes.

Correctness, which the paper does not report
  situation accuracy, emergency recall and false-alarm rate; correct response (emergency:
  LAND_AND_DEPLOY_RESCUE_KIT or ALERT_EMERGENCY_SERVICES recommended; normal: neither);
  person-count error against the true number of people in the scene; parse failures.
  For a system that assesses situations, a sample with no assessment (an unparseable
  reply) counts as wrong: not a caught emergency, not a correct "normal".
"""

import argparse
import glob
import json
import math
import os
import sys
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from agentic_uav.experiment.scenes import EMERGENCY, load

ALPHA = 0.05
CAR_MIN_CHARS = 50
EMERGENCY_ACTIONS = {"LAND_AND_DEPLOY_RESCUE_KIT", "ALERT_EMERGENCY_SERVICES"}


# ---- per-sample and per-system metrics (no scipy) --------------------------

def load_results(results_dir: str) -> Dict[str, List[dict]]:
    systems = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        by_scene = {r["scene_id"]: r for r in rows}           # last record per scene wins
        systems[os.path.basename(path)[:-len(".jsonl")]] = [by_scene[k] for k in sorted(by_scene)]
    # The rule-based baseline first, as in the paper's tables.
    return dict(sorted(systems.items(), key=lambda kv: (kv[0] != "rule", kv[0])))


def sample_metrics(rec: dict, truth) -> dict:
    o = rec["output"]
    actions = set(o.get("recommended_actions") or [])
    is_emergency = truth.label == EMERGENCY
    situation = o.get("situation")
    said_emergency = situation == EMERGENCY if situation else None
    return {
        "scene_id": rec["scene_id"],
        "emergency": is_emergency,
        "time_s": rec["processing_time_s"],
        "confidence": o.get("confidence"),
        "person_detected": bool(o.get("person_detected")),
        "arr": bool(o.get("recommended_actions")),
        "car": len(o.get("surrounding_features") or "") > CAR_MIN_CHARS,
        "assessed": said_emergency is not None,
        "said_emergency": said_emergency,
        "correct_response": bool(actions & EMERGENCY_ACTIONS) == is_emergency,
        "count_error": (o["persons_count"] - len(truth.people)) if o.get("persons_count") is not None else None,
        "true_people": len(truth.people),
        "yolo_count": len(rec["yolo"]["detections"]),
        "parse_ok": rec.get("llm", {}).get("parse_ok", True),
        "llm_time_s": rec.get("llm", {}).get("time_s"),
        "completion_tokens": rec.get("llm", {}).get("completion_tokens"),
        "thinking_chars": len(rec["llm"].get("thinking") or "") if "llm" in rec else None,
    }


def _mean_sd(xs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def _rate(flags: Sequence[bool]) -> Tuple[int, int]:
    return sum(bool(f) for f in flags), len(flags)


def system_summary(samples: List[dict]) -> dict:
    n = len(samples)
    emerg = [s for s in samples if s["emergency"]]
    normal = [s for s in samples if not s["emergency"]]
    # A system that assesses situations at all (the LLM systems) is scored on every sample;
    # a sample without an assessment (e.g. an unparseable reply) counts as a miss.
    assessed = any(s["assessed"] for s in samples)
    t_mean, t_sd = _mean_sd([s["time_s"] for s in samples])
    c_mean, c_sd = _mean_sd([s["confidence"] for s in samples])
    count_err = [s["count_error"] for s in samples if s["count_error"] is not None]
    yolo_err = [s["yolo_count"] - s["true_people"] for s in samples]
    out = {
        "n": n, "n_emergency": len(emerg), "n_normal": len(normal),
        "time_mean_s": t_mean, "time_sd_s": t_sd,
        "time_median_s": sorted(s["time_s"] for s in samples)[n // 2] if n else None,
        "confidence_mean": c_mean, "confidence_sd": c_sd,
        "persons_detected": _rate([s["person_detected"] for s in samples]),
        "arr": _rate([s["arr"] for s in samples]),
        "car": _rate([s["car"] for s in samples]),
        "parse_failures": sum(not s["parse_ok"] for s in samples),
        "correct_response": _rate([s["correct_response"] for s in samples]),
        "count_mae": (sum(abs(e) for e in count_err) / len(count_err)) if count_err else None,
        "count_bias": (sum(count_err) / len(count_err)) if count_err else None,
        "yolo_count_mae": sum(abs(e) for e in yolo_err) / n if n else None,
        "llm_time_mean_s": _mean_sd([s["llm_time_s"] for s in samples])[0],
        "completion_tokens_mean": _mean_sd([s["completion_tokens"] for s in samples])[0],
        "thinking_chars_mean": _mean_sd([s["thinking_chars"] for s in samples])[0],
    }
    if assessed:
        out["situation_accuracy"] = _rate([s["said_emergency"] is s["emergency"] for s in samples])
        out["emergency_recall"] = _rate([s["said_emergency"] is True for s in emerg])
        out["false_alarm_rate"] = _rate([s["said_emergency"] is True for s in normal])
        out["no_assessment"] = sum(not s["assessed"] for s in samples)
    else:
        out["situation_accuracy"] = out["emergency_recall"] = out["false_alarm_rate"] = None
        out["no_assessment"] = None
    return out


# ---- statistics (scipy) ----------------------------------------------------

def cohens_d(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    (ma, sa), (mb, sb) = _mean_sd(a), _mean_sd(b)
    if ma is None or mb is None or len(a) < 2 or len(b) < 2:
        return None
    pooled = math.sqrt(((len(a) - 1) * sa ** 2 + (len(b) - 1) * sb ** 2) / (len(a) + len(b) - 2))
    return (ma - mb) / pooled if pooled > 0 else None


def compare_continuous(groups: Dict[str, List[float]]) -> dict:
    from scipy import stats
    groups = {k: [x for x in v if x is not None] for k, v in groups.items()}
    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    if len(groups) < 2:
        return {"note": "fewer than two systems with data"}
    names = list(groups)
    res = {"normality": {}, "pairs": []}
    for k in names:
        v = groups[k]
        res["normality"][k] = None if len(set(v)) < 3 else float(stats.shapiro(v).pvalue)
    normal = all(p is not None and p > ALPHA for p in res["normality"].values())
    res["levene_p"] = float(stats.levene(*groups.values()).pvalue) if all(len(set(v)) > 1 for v in groups.values()) else None
    if len(names) >= 3:
        res["anova_p"] = float(stats.f_oneway(*groups.values()).pvalue)
        tk = stats.tukey_hsd(*groups.values())
    for i, j in combinations(range(len(names)), 2):
        a, b = groups[names[i]], groups[names[j]]
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        pair = {"a": names[i], "b": names[j],
                "mann_whitney_p": float(mw.pvalue),
                # Same sign as Cohen's d: positive when a's values tend to be higher.
                "rank_biserial_r": float(2 * mw.statistic / (len(a) * len(b)) - 1),
                "cohens_d": cohens_d(a, b)}
        if len(names) >= 3:
            pair["tukey_p"] = float(tk.pvalue[i, j])
        else:
            pair["welch_t_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        pair["test_used"] = ("tukey" if len(names) >= 3 else "welch_t") if normal else "mann_whitney"
        res["pairs"].append(pair)
    res["all_normal"] = normal
    return res


def compare_rates(rates: Dict[str, Tuple[int, int]]) -> dict:
    from scipy import stats
    rates = {k: v for k, v in rates.items() if v is not None and v[1] > 0}
    if len(rates) < 2:
        return {"note": "fewer than two systems with data"}
    table = [[k, n - k] for k, n in rates.values()]
    res = {"pairs": []}
    col = [sum(r[0] for r in table), sum(r[1] for r in table)]
    if 0 in col:
        res["chi2_p"] = 1.0       # every system identical (all yes or all no)
    else:
        res["chi2_p"] = float(stats.chi2_contingency(table).pvalue)
    names = list(rates)
    for i, j in combinations(range(len(names)), 2):
        t = [table[i], table[j]]
        if 0 in (t[0][0] + t[1][0], t[0][1] + t[1][1]):
            p, test = 1.0, "identical"
        else:
            expected_small = min(stats.contingency.expected_freq(t).ravel()) < 5
            p = float(stats.fisher_exact(t).pvalue if expected_small else stats.chi2_contingency(t).pvalue)
            test = "fisher" if expected_small else "chi2"
        res["pairs"].append({"a": names[i], "b": names[j], "p": p, "test": test})
    return res


# ---- report ----------------------------------------------------------------

def _pct(r) -> str:
    return "n/a" if r is None else f"{r[0]}/{r[1]} ({100.0 * r[0] / r[1]:.1f}%)" if r[1] else "0/0"


def _num(x, nd=3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _p(x) -> str:
    return "n/a" if x is None else ("<0.001" if x < 0.001 else f"{x:.3f}") + (" *" if x < ALPHA else "")


def report(summaries: Dict[str, dict], tests: Dict[str, dict]) -> str:
    names = list(summaries)
    lines = ["# Paper reproduction results", "",
             "## Paper metrics", "",
             "| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    rows = [("samples", lambda s: str(s["n"])),
            ("processing time, mean (SD) s", lambda s: f"{_num(s['time_mean_s'])} ({_num(s['time_sd_s'])})"),
            ("processing time, median s", lambda s: _num(s["time_median_s"])),
            ("detection confidence, mean (SD)", lambda s: f"{_num(s['confidence_mean'])} ({_num(s['confidence_sd'])})"),
            ("persons detected", lambda s: _pct(s["persons_detected"])),
            ("ARR (any action recommended)", lambda s: _pct(s["arr"])),
            (f"CAR (context > {CAR_MIN_CHARS} chars)", lambda s: _pct(s["car"]))]
    lines += [f"| {label} | " + " | ".join(f(summaries[n]) for n in names) + " |" for label, f in rows]
    lines += ["", "## Correctness against ground truth (not in the paper)", "",
              "| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    rows = [("situation accuracy", lambda s: _pct(s["situation_accuracy"])),
            ("emergency recall", lambda s: _pct(s["emergency_recall"])),
            ("false-alarm rate (normal scenes)", lambda s: _pct(s["false_alarm_rate"])),
            ("correct response", lambda s: _pct(s["correct_response"])),
            ("person count, mean abs error", lambda s: _num(s["count_mae"], 2)),
            ("person count, mean error (bias)", lambda s: _num(s["count_bias"], 2)),
            ("YOLO count, mean abs error", lambda s: _num(s["yolo_count_mae"], 2)),
            ("unparseable replies", lambda s: str(s["parse_failures"])),
            ("samples without an assessment (scored as wrong)",
             lambda s: "n/a" if s["no_assessment"] is None else str(s["no_assessment"]))]
    lines += [f"| {label} | " + " | ".join(f(summaries[n]) for n in names) + " |" for label, f in rows]
    lines += ["", "## LLM diagnostics", "",
              "| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    rows = [("LLM time, mean s", lambda s: _num(s["llm_time_mean_s"], 2)),
            ("completion tokens, mean", lambda s: _num(s["completion_tokens_mean"], 0)),
            ("thinking text, mean chars", lambda s: _num(s["thinking_chars_mean"], 0))]
    lines += [f"| {label} | " + " | ".join(f(summaries[n]) for n in names) + " |" for label, f in rows]
    lines += ["", "## Tests (alpha 0.05, * = significant)", ""]
    for metric, t in tests.items():
        lines.append(f"**{metric}**")
        if "note" in t:
            lines += [f"- {t['note']}", ""]
            continue
        if "normality" in t:
            lines.append("- Shapiro-Wilk p: " + ", ".join(f"{k} {_p(v)}" for k, v in t["normality"].items())
                         + f"; Levene p {_p(t.get('levene_p'))}" + (f"; ANOVA p {_p(t['anova_p'])}" if "anova_p" in t else ""))
            for pr in t["pairs"]:
                main = pr.get("tukey_p", pr.get("welch_t_p"))
                lines.append(f"- {pr['a']} vs {pr['b']}: {'Tukey' if 'tukey_p' in pr else 'Welch t'} p {_p(main)}, "
                             f"Mann-Whitney p {_p(pr['mann_whitney_p'])} (used: {pr['test_used']}), "
                             f"Cohen's d {_num(pr['cohens_d'], 2)}, rank-biserial r {_num(pr['rank_biserial_r'], 2)}")
        else:
            lines.append(f"- chi-square across systems p {_p(t['chi2_p'])}")
            for pr in t["pairs"]:
                lines.append(f"- {pr['a']} vs {pr['b']}: {pr['test']} p {_p(pr['p'])}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", default="/experiments/paper/scenes.json")
    ap.add_argument("--results", default="/experiments/paper/results")
    a = ap.parse_args()

    truth = {s.id: s for s in load(a.scenes)}
    systems = load_results(a.results)
    if not systems:
        print(f"[score] no results in {a.results}", file=sys.stderr)
        return 1
    samples = {name: [sample_metrics(r, truth[r["scene_id"]]) for r in rows if r["scene_id"] in truth]
               for name, rows in systems.items()}
    common = set.intersection(*(set(s["scene_id"] for s in v) for v in samples.values()))
    if any(len(v) != len(common) for v in samples.values()):
        print(f"[score] systems cover different scenes; scoring only the {len(common)} all have", file=sys.stderr)
        samples = {k: [s for s in v if s["scene_id"] in common] for k, v in samples.items()}
    summaries = {k: system_summary(v) for k, v in samples.items()}
    tests = {
        "processing time": compare_continuous({k: [s["time_s"] for s in v] for k, v in samples.items()}),
        "detection confidence": compare_continuous({k: [s["confidence"] for s in v] for k, v in samples.items()}),
        "persons detected": compare_rates({k: s["persons_detected"] for k, s in summaries.items()}),
        "ARR": compare_rates({k: s["arr"] for k, s in summaries.items()}),
        "CAR": compare_rates({k: s["car"] for k, s in summaries.items()}),
        "correct response": compare_rates({k: s["correct_response"] for k, s in summaries.items()}),
    }
    md = report(summaries, tests)
    with open(os.path.join(a.results, "summary.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(a.results, "summary.json"), "w") as f:
        json.dump({"systems": summaries, "tests": tests,
                   "samples": samples}, f, indent=1)
    print(md)
    print(f"\n[score] wrote {a.results}/summary.md and summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
