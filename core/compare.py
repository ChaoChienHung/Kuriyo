"""Compare settings over seeds with the tests in README "實驗設計與統計檢定".

Each setting is one or more run directories (one per seed) with per-user ranks
(validation/<set>/best-ranks.json, or test/<set>/ranks.json with --split test; runs
evaluated before ranks existed get them from STAGE=groups). A ranks file may be given
instead of a run directory, such as TIGER's validation/<set>/best-beam100-ranks.json
from STAGE=beams. Every other setting is compared with the first:

    python -m core.compare --metric ndcg@10 \\
        --setting paper Recall/SASRec/baseline/runs/sasrec-paper-beauty-{v1,s43,s44} \\
        --setting ce Recall/SASRec/baseline/runs/sasrec-paperce-beauty-{v1,s43,s44}

Per setting: mean ± sample sd over seeds and the t 95% CI. Per comparison: the paired
per-user test (each user's metric averaged over seeds; paired t and a 10,000-resample
paired bootstrap) and Welch's t over seed values, Holm-adjusted across comparisons, and
the README verdict: "significant" when both adjusted p < 0.05 and the mean gap exceeds
the larger seed sd, "tendency" when at least one test has adjusted p < 0.05 but not all
conditions hold, "no difference shown" when neither test does, "single seed" when Welch
cannot run.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from core.stats import holm, paired_bootstrap, paired_t, summary, welch_t

ALPHA = 0.05


def metric_values(ranks, metric):
    """Per-request values of recall@k or ndcg@k from 1-based ranks (0 = not in the list)."""
    name, _, k = metric.partition("@")
    if name not in ("recall", "ndcg") or not k.isdigit() or int(k) < 1:
        raise ValueError("metric must be recall@K or ndcg@K")
    ranks = np.asarray(ranks)
    hit = (ranks > 0) & (ranks <= int(k))
    if name == "recall":
        return hit.astype(float)
    return np.where(hit, 1.0 / np.log2(np.maximum(ranks, 1) + 1.0), 0.0)


def ranks_path(run, split, eval_set=None):
    base = Path(run) / split
    sets = [eval_set] if eval_set else sorted(path.name for path in base.glob("*") if path.is_dir())
    if len(sets) != 1:
        raise ValueError("{}: expected one evaluation set under {}, found {}; use --set".format(run, base, sets))
    return base / sets[0] / ("best-ranks.json" if split == "validation" else "ranks.json")


def load_run(run, split, metric, eval_set=None):
    """run is a run directory, or one ranks file inside it (for example a TIGER
    validation/<set>/best-beam100-ranks.json); a file is used as given."""
    run = Path(run)
    if run.is_file():
        path = run
        root = next((parent for parent in path.parents if (parent / "settings.json").is_file()), path.parent)
        recorded = path.with_name(path.name.replace("-ranks.json", ".json"))
    else:
        path = ranks_path(run, split, eval_set)
        root = run
        recorded = path.parent / ("best.json" if split == "validation" else "metrics.json")
    if not path.is_file():
        raise ValueError("{} missing; run STAGE=groups for this run first".format(path))
    record = json.loads(path.read_text())
    _, _, k = metric.partition("@")
    if int(k) > record["max_rank"]:
        raise ValueError("{}: ranks only go to {}".format(path, record["max_rank"]))
    settings_path = root / "settings.json"
    seed = json.loads(settings_path.read_text())["run"].get("seed") if settings_path.is_file() else None
    values = metric_values(record["ranks"], metric)
    difference = None
    if recorded != path and recorded.is_file():
        stored = json.loads(recorded.read_text()).get(metric)
        if stored is not None:
            difference = abs(float(values.mean()) - stored)
    return {"run": str(run), "seed": seed, "users": record["users"], "values": values,
            "mean": float(values.mean()), "recorded_difference": difference}


def compare(settings, split="validation", metric="ndcg@10", eval_set=None, resamples=10000, seed=0):
    """settings: list of (name, [run dirs]). Returns a JSON-serialisable report."""
    if len(settings) < 1:
        raise ValueError("give at least one setting")
    loaded = {name: [load_run(run, split, metric, eval_set) for run in runs] for name, runs in settings}
    reference_users = None
    for name, runs in loaded.items():
        for run in runs:
            if reference_users is None:
                reference_users = run["users"]
            elif run["users"] != reference_users:
                raise ValueError("{} evaluates different users or order than the first run; "
                                 "compare runs on the same split only".format(run["run"]))
    report = {"split": split, "metric": metric, "users": len(reference_users), "settings": [], "comparisons": [],
              "warnings": []}
    seeds = {name: sorted(str(run["seed"]) for run in runs) for name, runs in loaded.items()}
    if len({tuple(value) for value in seeds.values()}) > 1:
        report["warnings"].append("settings use different seed sets: {}".format(seeds))
    for name, runs in loaded.items():
        for run in runs:
            if run["recorded_difference"] is not None and run["recorded_difference"] > 1e-9:
                report["warnings"].append("{}: {} from ranks differs from the recorded metric by {:.2e}".format(
                    run["run"], metric, run["recorded_difference"]))
        report["settings"].append({"name": name, "runs": [run["run"] for run in runs],
                                   "seeds": [run["seed"] for run in runs],
                                   "values": [run["mean"] for run in runs],
                                   **summary([run["mean"] for run in runs])})
    per_user = {name: np.mean([run["values"] for run in runs], axis=0) for name, runs in loaded.items()}
    base_name = settings[0][0]
    base = report["settings"][0]
    for name, _ in settings[1:]:
        other = next(entry for entry in report["settings"] if entry["name"] == name)
        differences = per_user[name] - per_user[base_name]
        report["comparisons"].append({
            "setting": name, "baseline": base_name, "difference": other["mean"] - base["mean"],
            "paired_t": paired_t(differences),
            "paired_bootstrap": paired_bootstrap(differences, resamples, seed),
            "welch": welch_t(other["values"], base["values"])})
    comparisons = report["comparisons"]
    for key, adjusted in (("paired_t", holm([c["paired_t"]["p"] for c in comparisons])),
                          ("paired_bootstrap", holm([c["paired_bootstrap"]["p"] for c in comparisons])),
                          ("welch", holm([c["welch"]["p"] if c["welch"] else None for c in comparisons]))):
        for comparison, p in zip(comparisons, adjusted):
            if comparison[key] is not None:
                comparison[key]["p_holm"] = p
    for comparison in comparisons:
        other = next(entry for entry in report["settings"] if entry["name"] == comparison["setting"])
        comparison["verdict"] = verdict(comparison, base, other)
    return report


def verdict(comparison, base, other):
    if comparison["welch"] is None:
        return "single seed: no claim"
    user_p = max(comparison["paired_t"]["p_holm"], comparison["paired_bootstrap"]["p_holm"])
    conditions = [user_p < ALPHA, comparison["welch"]["p_holm"] < ALPHA,
                  abs(comparison["difference"]) > max(base["sd"], other["sd"])]
    direction = "better" if comparison["difference"] > 0 else "worse"
    if all(conditions):
        return "significantly {}".format(direction)
    if conditions[0] or conditions[1]:
        return "tendency {} (paired p<0.05: {}, Welch p<0.05: {}, gap > seed sd: {})".format(direction, *conditions)
    return "no difference shown"


def _number(value, digits=5):
    return "—" if value is None else "{:.{}f}".format(value, digits)


def _p(value):
    return "—" if value is None else "{:.2g}".format(value)


def markdown(report):
    lines = ["{} on {} ({} users)".format(report["metric"], report["split"], report["users"]), "",
             "| setting | seeds | mean ± sd | 95% CI |", "|---|---|---|---|"]
    for entry in report["settings"]:
        ci = "—" if entry["ci"] is None else "[{}, {}]".format(_number(entry["ci"][0]), _number(entry["ci"][1]))
        sd = "" if entry["sd"] is None else " ± " + _number(entry["sd"])
        lines.append("| {} | {} | {}{} | {} |".format(entry["name"], ", ".join(map(str, entry["seeds"])),
                                                     _number(entry["mean"]), sd, ci))
    if report["comparisons"]:
        lines += ["", "| vs {} | difference | paired t p (Holm) | bootstrap 95% CI | bootstrap p (Holm) "
                  "| Welch p (Holm) | verdict |".format(report["comparisons"][0]["baseline"]),
                  "|---|---|---|---|---|---|---|"]
        for c in report["comparisons"]:
            boot = c["paired_bootstrap"]
            lines.append("| {} | {:+.5f} | {} ({}) | [{:+.5f}, {:+.5f}] | {} ({}) | {} | {} |".format(
                c["setting"], c["difference"], _p(c["paired_t"]["p"]), _p(c["paired_t"]["p_holm"]),
                boot["ci"][0], boot["ci"][1], _p(boot["p"]), _p(boot["p_holm"]),
                "—" if c["welch"] is None else "{} ({})".format(_p(c["welch"]["p"]), _p(c["welch"]["p_holm"])),
                c["verdict"]))
    for warning in report["warnings"]:
        lines.append("\nwarning: " + warning)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", nargs="+", action="append", required=True, metavar=("NAME", "RUN"),
                        help="a name followed by its run directories (one per seed); the first is the baseline")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--metric", default="ndcg@10", help="recall@K or ndcg@K (K <= recorded max_rank)")
    parser.add_argument("--set", dest="eval_set", help="evaluation set folder, when a run has several")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="also write the full report here")
    args = parser.parse_args(argv)
    if any(len(setting) < 2 for setting in args.setting):
        parser.error("each --setting needs a name and at least one run directory")
    names = [setting[0] for setting in args.setting]
    if len(set(names)) != len(names):
        parser.error("setting names must be unique")
    try:
        report = compare([(setting[0], setting[1:]) for setting in args.setting], args.split, args.metric,
                         args.eval_set, args.resamples, args.bootstrap_seed)
    except ValueError as error:
        parser.error(str(error))
    print(markdown(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
