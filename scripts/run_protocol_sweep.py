"""
Protocol sweep: does evaluation protocol or architecture drive AUROC on CTU-UHB?

WHY THIS EXISTS
---------------
Measured 2026-09-03 with 19 hand-crafted features and a fixed classifier, on
data/processed_clinical/ (547 patients, patient-level label, no horizon rule):

    protocol                real labels   RANDOM labels
    random-window split        0.898          0.909
    patient-grouped split      0.625          0.517

A protocol that scores 0.909 on labels carrying no clinical information is not
measuring physiology. The mechanism is patient re-identification: at a 2.5-min
stride consecutive 20-min windows share 87.5% of their samples, and a held-out
window's nearest neighbour in feature space is from the same labour 74.6% of
the time (423x chance).

This script asks whether the same holds for DEEP models across a 19x capacity
range, which the feature baseline cannot answer. The prediction is that under
window_random, AUROC rises with parameter count (bigger models memorise better),
while under stratified_group it stays flat or falls (bigger models overfit 76
positive patients). If those lines cross, architecture progress on this
benchmark is tracking memorisation capacity, not physiological modelling.

FRAMING -- READ BEFORE CITING ANY PAPER
---------------------------------------
This is a REPLICATION and negative-control study, NOT an audit of specific
authors. Dang et al. (2026) explicitly report Stratified Group K-Fold with all
windows of a patient held in one fold -- see this repo's own trainer docstring,
which replicates that protocol. The foundation-model paper splits 441/56/55 at
recording level and evaluates on 55 held-out recordings. NEITHER is accused of
window-level leakage here, and the horizon/time confound was this project's own
pipeline bug, not theirs (docs/literature_preprocessing_comparison.md).

What this sweep supports is the narrower, defensible claim: window-level
partitioning of overlapping CTG windows is a negative-control failure for this
database, so results must be reported under patient-grouped splits with
patient-clustered confidence intervals.

USAGE
-----
    # the headline 2x8 grid (honest vs negative-control protocol)
    python scripts/run_protocol_sweep.py --data_dir data/processed_clinical/

    # add the random-label control (doubles the runtime)
    python scripts/run_protocol_sweep.py --permute

    # cheap first pass
    python scripts/run_protocol_sweep.py --epochs 15 --encoders cnn1d,crossformer

Cells are skipped if their results JSON already exists, so the sweep is
resumable after an interrupt. Each cell gets its own checkpoint directory --
the trainer's test-set ensemble globs checkpoints, so a shared directory would
silently mix one cell's folds into another cell's test score.
"""
import argparse
import json
import os
import subprocess
import sys
import time

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# Ordered by parameter count so the capacity trend reads down the table.
DEFAULT_ENCODERS = ["cnn1d", "bilstm", "gru", "tcn", "mslstm",
                    "patchctg", "patchtst", "crossformer"]
PROTOCOLS = ["stratified_group", "window_random"]


def cell_name(encoder, protocol, permute):
    return f"{encoder}__{protocol}{'__permuted' if permute else ''}"


def run_cell(args, encoder, protocol, permute):
    name = cell_name(encoder, protocol, permute)
    out_json = os.path.join(args.results_dir, f"{name}.json")
    if os.path.exists(out_json) and not args.force:
        print(f"  [skip] {name} -- results exist")
        return out_json, True

    ckpt_dir = os.path.join(args.checkpoint_root, name)
    os.makedirs(ckpt_dir, exist_ok=True)
    # A stale checkpoint from an earlier attempt would be picked up by the
    # trainer's ensemble glob and blended into this cell's test score.
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pth"):
            os.remove(os.path.join(ckpt_dir, f))

    cmd = [sys.executable, "-m", "src.models.train_ctg_crossformer",
           "--data_dir", args.data_dir,
           "--encoder", encoder,
           "--fold_mode", protocol,
           "--checkpoint_dir", ckpt_dir,
           "--results_json", out_json,
           "--target", args.target,
           "--class_weight", args.class_weight,
           "--early_stop_mode", args.early_stop_mode,
           "--seed", str(args.seed)]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if permute:
        cmd += ["--permute_labels"]

    print(f"  [run ] {name}")
    t0 = time.time()
    log_path = os.path.join(args.results_dir, f"{name}.log")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=BASE, stdout=log,
                              stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    if proc.returncode != 0 or not os.path.exists(out_json):
        print(f"  [FAIL] {name} (exit {proc.returncode}, {dt/60:.1f} min) -- see {log_path}")
        return None, False
    print(f"  [done] {name} in {dt/60:.1f} min")
    return out_json, True


def load(results_dir, encoder, protocol, permute):
    p = os.path.join(results_dir, f"{cell_name(encoder, protocol, permute)}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def fmt(d, key):
    if d is None or key not in d.get("cv", {}):
        return "--"
    return f"{d['cv'][key]['mean']:.4f}"


def report(args, encoders):
    lines = []
    add = lines.append
    add("# Protocol sweep -- does the split rule or the architecture drive AUROC?\n")
    add(f"Data: `{args.data_dir}` | target `{args.target}` | "
        f"{args.epochs or 'config'} epochs | seed {args.seed}\n")
    add("`window_random` is a NEGATIVE CONTROL, not a result. It is reported only "
        "to quantify how much a window-level split inflates this benchmark.\n")

    add("\n## Window-level AUROC (5-fold CV mean)\n")
    add("| encoder | params | patient-grouped | window-random | inflation |")
    add("|---|---:|---:|---:|---:|")
    for e in encoders:
        g = load(args.results_dir, e, "stratified_group", False)
        w = load(args.results_dir, e, "window_random", False)
        n_p = g["n_params"] if g else (w["n_params"] if w else None)
        n_str = f"{n_p:,}" if n_p else "--"
        infl = ("--" if not (g and w) else
                f"{w['cv']['auroc']['mean'] - g['cv']['auroc']['mean']:+.4f}")
        add(f"| {e} | {n_str} | {fmt(g, 'auroc')} | {fmt(w, 'auroc')} | {infl} |")

    add("\n## Patient-level AUROC -- the clinically meaningful number\n")
    add("| encoder | grouped (max-agg) | grouped (mean-agg) | window-random (max-agg) |")
    add("|---|---:|---:|---:|")
    for e in encoders:
        g = load(args.results_dir, e, "stratified_group", False)
        w = load(args.results_dir, e, "window_random", False)
        add(f"| {e} | {fmt(g, 'auroc_pat_max')} | {fmt(g, 'auroc_pat_mean')} | "
            f"{fmt(w, 'auroc_pat_max')} |")

    if args.permute:
        add("\n## Negative control -- RANDOM labels (a sound protocol must give ~0.50)\n")
        add("| encoder | patient-grouped | window-random |")
        add("|---|---:|---:|")
        for e in encoders:
            gp = load(args.results_dir, e, "stratified_group", True)
            wp = load(args.results_dir, e, "window_random", True)
            add(f"| {e} | {fmt(gp, 'auroc')} | {fmt(wp, 'auroc')} |")

    add("\n## Reference -- 19 features + logistic regression, identical data\n")
    add("| protocol | real labels | random labels |")
    add("|---|---:|---:|")
    add("| window-random | 0.8979 | 0.9094 |")
    add("| patient-grouped | 0.6254 | 0.5170 |")
    add("\nPatient-grouped patient-level feature baseline: **0.7290 +/- 0.045** "
        "(pooled 5x5 CV, 547 patients).\n")

    out = os.path.join(args.results_dir, "SWEEP_REPORT.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/processed_clinical/")
    ap.add_argument("--results_dir", default="results/protocol_sweep/")
    ap.add_argument("--checkpoint_root", default="checkpoints/protocol_sweep/")
    ap.add_argument("--encoders", default=",".join(DEFAULT_ENCODERS),
                    help="comma-separated; default is all 8, ordered by parameter count")
    ap.add_argument("--protocols", default=",".join(PROTOCOLS))
    ap.add_argument("--permute", action="store_true",
                    help="also run the random-label negative control (doubles runtime)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--target", default="y_primary")
    ap.add_argument("--class_weight", default="none",
                    choices=["none", "inverse_freq", "sqrt_inverse_freq"])
    ap.add_argument("--early_stop_mode", default="nested",
                    choices=["outer_best", "nested"],
                    help="nested (default here) gives the UNBIASED estimate. The trainer's "
                         "own default is outer_best for backward compatibility, which selects "
                         "the epoch on the reported fold and inflates CV AUROC by ~0.077 -- "
                         "not something to layer on top of a protocol study.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="re-run cells that already have results")
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_root, exist_ok=True)

    if not args.report_only:
        permutes = [False, True] if args.permute else [False]
        cells = [(e, p, q) for e in encoders for p in protocols for q in permutes]
        print(f"{len(cells)} cells: {len(encoders)} encoders x {len(protocols)} protocols"
              f"{' x 2 label conditions' if args.permute else ''}\n")
        t0 = time.time()
        failed = []
        for i, (e, p, q) in enumerate(cells, 1):
            print(f"[{i}/{len(cells)}]", end=" ")
            _, ok = run_cell(args, e, p, q)
            if not ok:
                failed.append(cell_name(e, p, q))
        print(f"\nSweep finished in {(time.time() - t0) / 60:.1f} min")
        if failed:
            print(f"{len(failed)} cell(s) FAILED: {', '.join(failed)}")

    report(args, encoders)


if __name__ == "__main__":
    main()
