"""
Self-contained evidence script for the evaluation-protocol analysis.

Every number quoted in docs/literature_forensic_audit.md and
docs/positioning_and_claims.md is produced here. It needs NO network access and
NO GPU -- only data/processed_clinical/ and scikit-learn -- so the whole
evidence base is reproducible offline, on any machine, indefinitely.

    python scripts/audit_evaluation_protocol.py
    python scripts/audit_evaluation_protocol.py --probes protocol,identity
    python scripts/audit_evaluation_protocol.py --out docs/evaluation_protocol_evidence.md

WHAT EACH PROBE ESTABLISHES
---------------------------
protocol        The 2x2: {patient-grouped, window-random} x {real, random labels}.
                A window-random split scores ~0.91 on labels containing no
                clinical information at all. This is a NEGATIVE-CONTROL FAILURE
                of the protocol, not a finding about any published paper --
                see the framing note below.
identity        The mechanism behind it: a held-out window's nearest neighbour
                in feature space belongs to the same labour ~75% of the time.
label           Rebuilds this project's OLD horizon label on the clean data and
                shows it is ~88% predictable from elapsed time alone. The
                horizon rule was this project's own invention, not inherited
                from any paper (docs/literature_preprocessing_comparison.md).
aggregation     Window-level vs patient-level scoring on identical predictions.
target          Does moving the pH<=7.15 cut, or switching outcome, help? (No.)
cohort          What excluding borderline or low-quality patients buys.
representation  Do patient-level / trajectory features beat per-window+aggregate?
clustering      How badly an independence assumption understates the CI when
                observations are overlapping windows clustered in patients.

FRAMING -- READ BEFORE CITING ANY PAPER
---------------------------------------
Dang et al. (2026) explicitly report 5-fold Stratified Group K-Fold with all
windows of a patient confined to one fold. The 2026 foundation model splits
552 recordings 441/56/55 at the RECORDING level and reports AUC on 55 held-out
recordings. NEITHER exhibits the window-level leakage this script demonstrates,
and neither uses a horizon-style label. The `protocol` probe is a statement
about what this benchmark PERMITS, not an accusation about what anyone did.

The defensible claims are narrower and are the ones the docs make: report
patient-grouped splits, patient-level units, and patient-clustered intervals.
"""
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from sklearn.ensemble import RandomForestClassifier                  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge           # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score   # noqa: E402
from sklearn.model_selection import StratifiedKFold                  # noqa: E402
from sklearn.neighbors import NearestNeighbors                       # noqa: E402
from sklearn.pipeline import make_pipeline                           # noqa: E402
from sklearn.preprocessing import StandardScaler                     # noqa: E402

PROBES = ["protocol", "identity", "label", "aggregation", "target",
          "cohort", "representation", "clustering"]
N_REPEATS = 5
N_FOLDS = 5


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_all(data_dir):
    """Window features + every per-window annotation, pooled over all splits.

    The 19 features are the 8 FIGO features the pipeline computes plus the 11
    extended ones (src/knowledge/extended_features.py). Pooling train/val/test
    is deliberate: these probes use their own cross-validation, so the shipped
    split is irrelevant and pooling gives the full 547-patient cohort.
    """
    import torch
    F, y, pid, t_end, ph, qual, figo = [], [], [], [], [], [], []
    for split in ("train", "val", "test"):
        p = os.path.join(data_dir, f"{split}_dataset.pt")
        if not os.path.exists(p):
            sys.exit(f"[ABORT] missing {p}. Build it with "
                     f"src/preprocessing/pipeline_clinical.py first.")
        d = torch.load(p, weights_only=False)
        ext_p = os.path.join(data_dir, f"{split}_extended_features.npy")
        if not os.path.exists(ext_p):
            sys.exit(f"[ABORT] missing {ext_p}.")
        F.append(np.hstack([d["y_features"].numpy(), np.load(ext_p)]))
        y.append(d["y_primary"].numpy())
        pid.append(np.array([m[0] for m in d["metadata"]]))
        t_end.append(d["w_minutes_before_end"].numpy())
        ph.append(d["y_ph"].numpy())
        qual.append(d["w_quality"].numpy())
        figo.append(d["y_figo"].numpy())
    D = dict(F=np.vstack(F), y=np.concatenate(y), pid=np.concatenate(pid),
             t_end=np.concatenate(t_end), ph=np.concatenate(ph),
             qual=np.concatenate(qual), figo=np.concatenate(figo))
    D["patients"] = np.array(sorted(set(D["pid"])))
    D["pidx"] = {p: np.where(D["pid"] == p)[0] for p in D["patients"]}
    D["plab"] = np.array([int(D["y"][D["pidx"][p]].max()) for p in D["patients"]])
    D["pph"] = np.array([D["ph"][D["pidx"][p]][0] for p in D["patients"]])
    return D


def lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def grouped_folds(D, labels=None, repeats=N_REPEATS):
    """Yield (train_window_mask, test_patients) with patients never split."""
    lab = D["plab"] if labels is None else labels
    for rep in range(repeats):
        for tr, te in StratifiedKFold(N_FOLDS, shuffle=True,
                                      random_state=rep).split(D["patients"], lab):
            trS = set(D["patients"][tr])
            yield np.array([q in trS for q in D["pid"]]), D["patients"][te]


def permuted_patient_labels(D, seed=0):
    """Shuffle the outcome across patients: prevalence kept, meaning removed."""
    rng = np.random.default_rng(seed)
    fake = dict(zip(D["patients"], rng.permutation(D["plab"])))
    return np.array([fake[q] for q in D["pid"]]), fake


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------
def probe_protocol(D, out):
    out("## protocol -- the 2x2 negative control\n")
    out("Identical data, features and classifier throughout. Only the split "
        "rule and the labels change.\n")
    yw_fake, fake = permuted_patient_labels(D)
    fake_plab = np.array([fake[p] for p in D["patients"]])
    rows = []
    for model_name, mk in (("LogReg", lr),
                           ("RandomForest", lambda: RandomForestClassifier(
                               n_estimators=300, min_samples_leaf=3,
                               class_weight="balanced", random_state=0, n_jobs=-1))):
        for lab_name, yw, plb in (("real", D["y"], D["plab"]),
                                  ("RANDOM", yw_fake, fake_plab)):
            grp = []
            for m, _ in grouped_folds(D, labels=plb, repeats=3):
                if len(np.unique(yw[m])) < 2:
                    continue
                c = mk().fit(D["F"][m], yw[m])
                grp.append(roc_auc_score(yw[~m], c.predict_proba(D["F"][~m])[:, 1]))
            win = []
            for rep in range(3):
                for tr, te in StratifiedKFold(N_FOLDS, shuffle=True,
                                              random_state=rep).split(D["F"], yw):
                    c = mk().fit(D["F"][tr], yw[tr])
                    win.append(roc_auc_score(yw[te], c.predict_proba(D["F"][te])[:, 1]))
            rows.append((model_name, lab_name, np.mean(grp), np.mean(win)))
    out("| model | labels | patient-grouped | window-random | inflation |")
    out("|---|---|---:|---:|---:|")
    for m, l, g, w in rows:
        out(f"| {m} | {l} | {g:.4f} | {w:.4f} | {w - g:+.4f} |")
    out("\nA window-random split scores far above chance on RANDOM labels. Low-capacity "
        "models barely benefit; high-capacity models benefit enormously -- which is the "
        "signature of memorisation, not of a weak signal being amplified.\n")


def probe_identity(D, out):
    out("## identity -- the mechanism\n")
    F = StandardScaler().fit_transform(D["F"])
    n = len(D["pid"])
    cnt = {p: len(D["pidx"][p]) for p in D["patients"]}
    chance = np.mean([(cnt[p] - 1) / (n - 1) for p in D["pid"]])
    _, idx = NearestNeighbors(n_neighbors=6).fit(F).kneighbors(F)
    same1 = np.mean(D["pid"][idx[:, 1]] == D["pid"])
    same5 = np.mean([(D["pid"][idx[i, 1:6]] == D["pid"][i]).any() for i in range(n)])
    out(f"{n} windows, {len(D['patients'])} patients. Chance that a random other "
        f"window shares your patient: {chance:.4f}\n")
    out("| retrieval | rate | vs chance |")
    out("|---|---:|---:|")
    out(f"| nearest neighbour is same patient | {same1:.4f} | {same1 / chance:.0f}x |")
    out(f"| same patient within top-5 | {same5:.4f} | {same5 / chance / 5:.0f}x |")
    out("\nWindows of one labour cluster tightly. Under a window-random split those "
        "clusters straddle train and test, so the model retrieves the patient and the "
        "label comes along for free.\n")


def probe_label(D, out):
    out("## label -- reconstructing this project's OLD horizon rule\n")
    # Old rule: positive iff acidotic AND window starts in the last 30 min.
    # end = start + 4800 (20 min at 4 Hz), so start >= seg_len - 7200 is
    # equivalent to minutes_before_end <= 10.
    y_h = (D["y"] * (D["t_end"] <= 10.0)).astype(int)
    out(f"patient label positives: {int(D['y'].sum())} ({100 * D['y'].mean():.1f}% of windows) | "
        f"horizon label positives: {int(y_h.sum())} ({100 * y_h.mean():.1f}%)\n")
    out("| label | window AUROC (grouped) | AUROC from ELAPSED TIME alone |")
    out("|---|---:|---:|")
    for nm, yw in (("patient (clean, current)", D["y"]), ("horizon (old pipeline)", y_h)):
        a = []
        for m, _ in grouped_folds(D, repeats=3):
            if len(np.unique(yw[m])) < 2:
                continue
            c = lr().fit(D["F"][m], yw[m])
            a.append(roc_auc_score(yw[~m], c.predict_proba(D["F"][~m])[:, 1]))
        t_auc = roc_auc_score(yw, -D["t_end"])
        out(f"| {nm} | {np.mean(a):.4f} | {t_auc:.4f} |")
    out("\nA clock with no access to the heart-rate signal beats a trained model on the "
        "horizon label, and scores chance on the clean one. The confound lived entirely "
        "in the label. That rule was this project's own; no reviewed paper uses it.\n")


def probe_aggregation(D, out):
    out("## aggregation -- window-level vs patient-level, identical predictions\n")
    res = {k: [] for k in ("win", "mean", "max", "p90", "ap_max")}
    for m, teP in grouped_folds(D):
        c = lr().fit(D["F"][m], D["y"][m])
        s = c.predict_proba(D["F"])[:, 1]
        res["win"].append(roc_auc_score(D["y"][~m], s[~m]))
        pl = np.array([int(D["y"][D["pidx"][p]].max()) for p in teP])
        for nm, fn in (("mean", np.mean), ("max", np.max),
                       ("p90", lambda v: np.percentile(v, 90))):
            ps = np.array([fn(s[D["pidx"][p]]) for p in teP])
            res[nm].append(roc_auc_score(pl, ps))
            if nm == "max":
                res["ap_max"].append(average_precision_score(pl, ps))
    out("| scoring unit | AUROC |")
    out("|---|---:|")
    out(f"| window-level | {np.mean(res['win']):.4f} +/- {np.std(res['win']):.4f} |")
    for nm in ("mean", "p90", "max"):
        out(f"| patient-level ({nm}-aggregated) | {np.mean(res[nm]):.4f} +/- {np.std(res[nm]):.4f} |")
    out(f"\nPatient-level AUPRC (max-agg): {np.mean(res['ap_max']):.4f}. "
        "The label is constant within a patient, so the patient is the clinical unit; "
        "window-level scoring both understates performance and overstates the effective "
        "sample size.\n")


def _patient_cv(D, plab, score_fn):
    aucs, aps = [], []
    for m, teP in grouped_folds(D, labels=plab):
        ylab = {p: plab[i] for i, p in enumerate(D["patients"])}
        yw = np.array([ylab[q] for q in D["pid"]])
        if len(np.unique(yw[m])) < 2:
            continue
        s = score_fn(m, yw)
        ps = np.array([s[D["pidx"][p]].max() for p in teP])
        pl = np.array([ylab[p] for p in teP])
        if len(np.unique(pl)) < 2:
            continue
        aucs.append(roc_auc_score(pl, ps))
        aps.append(average_precision_score(pl, ps))
    return np.mean(aucs), np.std(aucs), np.mean(aps)


def probe_target(D, out):
    out("## target -- does moving the pH cut or switching outcome help?\n")
    import torch
    bd, ap5 = [], []
    for split in ("train", "val", "test"):
        d = torch.load(os.path.join(D["_dir"], f"{split}_dataset.pt"), weights_only=False)
        bd.append(d["y_bdecf"].numpy())
        ap5.append(d["y_apgar5"].numpy())
    bd_w, ap_w = np.concatenate(bd), np.concatenate(ap5)
    pbd = np.array([bd_w[D["pidx"][p]][0] for p in D["patients"]])
    pap = np.array([ap_w[D["pidx"][p]][0] for p in D["patients"]])
    pbd_f = np.nan_to_num(pbd, nan=-99.0)

    targets = {}
    for t in (7.00, 7.05, 7.10, 7.15, 7.20, 7.25):
        targets[f"pH <= {t:.2f}"] = (D["pph"] <= t).astype(int)
    targets["BDecf >= 8"] = (pbd_f >= 8).astype(int)
    targets["BDecf >= 12"] = (pbd_f >= 12).astype(int)
    targets["Apgar5 < 7"] = (pap < 7).astype(int)
    targets["composite (7.05|BD12|Ap7)"] = (
        (D["pph"] <= 7.05) | (pbd_f >= 12) | (pap < 7)).astype(int)
    targets["pH<=7.15 AND BD>=8"] = ((D["pph"] <= 7.15) & (pbd_f >= 8)).astype(int)

    out("| target | n pos | prevalence | patient AUROC | AUPRC |")
    out("|---|---:|---:|---:|---:|")
    for nm, plab in targets.items():
        if plab.sum() < 12:
            out(f"| {nm} | {plab.sum()} | {100 * plab.mean():.1f}% | too few to CV | -- |")
            continue
        a, sd, apc = _patient_cv(D, plab, lambda m, yw: lr().fit(
            D["F"][m], yw[m]).predict_proba(D["F"])[:, 1])
        star = "  <-- current" if nm == "pH <= 7.15" else ""
        out(f"| {nm}{star} | {plab.sum()} | {100 * plab.mean():.1f}% | "
            f"{a:.4f} +/- {sd:.4f} | {apc:.4f} |")

    out("\n### continuous-pH regression as the training signal\n")
    out("| scored against | patient AUROC |")
    out("|---|---:|")
    for tname in ("pH <= 7.05", "pH <= 7.10", "pH <= 7.15"):
        a, sd, _ = _patient_cv(D, targets[tname], lambda m, yw: -make_pipeline(
            StandardScaler(), Ridge(alpha=10.0)).fit(D["F"][m], D["ph"][m]).predict(D["F"]))
        out(f"| {tname} | {a:.4f} +/- {sd:.4f} |")
    out("\nFlat across every clinically plausible cut. The threshold is not the "
        "constraint; the number of positive patients is.\n")


def probe_cohort(D, out):
    out("## cohort -- what excluding patients buys (train AND test)\n")
    pq = np.array([D["qual"][D["pidx"][p]].mean() for p in D["patients"]])
    configs = [
        ("all 547 (honest baseline)", np.ones(len(D["patients"]), bool)),
        ("drop pH 7.10-7.20 (ambiguous)", ~((D["pph"] > 7.10) & (D["pph"] < 7.20))),
        ("drop pH 7.05-7.25 (wide band)", ~((D["pph"] > 7.05) & (D["pph"] < 7.25))),
        ("top 75% signal quality", pq >= np.percentile(pq, 25)),
        ("top 50% signal quality", pq >= np.percentile(pq, 50)),
    ]
    out("| cohort | n | n pos | patient AUROC |")
    out("|---|---:|---:|---:|")
    for nm, mask in configs:
        sub = D["patients"][mask]
        plab = (D["pph"][mask] <= 7.15).astype(int)
        if plab.sum() < 10 or (len(plab) - plab.sum()) < 10:
            out(f"| {nm} | {len(sub)} | {plab.sum()} | degenerate |")
            continue
        ylab = dict(zip(sub, plab))
        aucs = []
        for rep in range(N_REPEATS):
            for tr, te in StratifiedKFold(N_FOLDS, shuffle=True,
                                          random_state=rep).split(sub, plab):
                trS = set(sub[tr])
                m = np.array([q in trS for q in D["pid"]])
                yw = np.array([ylab.get(q, -1) for q in D["pid"]])
                mm = m & (yw >= 0)
                s = lr().fit(D["F"][mm], yw[mm]).predict_proba(D["F"])[:, 1]
                ps = np.array([s[D["pidx"][p]].max() for p in sub[te]])
                aucs.append(roc_auc_score(np.array([ylab[p] for p in sub[te]]), ps))
        out(f"| {nm} | {len(sub)} | {plab.sum()} | {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} |")
    out("\nExcluding borderline pH from BOTH train and test is the only real lever, and "
        "it is an easier question rather than a better model. Quality filtering hurts.\n")


def probe_representation(D, out):
    out("## representation -- do patient-level / trajectory features help?\n")
    ordr = {p: np.argsort(-D["t_end"][D["pidx"][p]]) for p in D["patients"]}

    def build(kind):
        rows = []
        for p in D["patients"]:
            idx = D["pidx"][p][ordr[p]]
            Fp, g = D["F"][idx], D["figo"][idx]
            n, parts = len(Fp), []
            if "max" in kind:
                parts.append(Fp.max(0))
            if "mean" in kind:
                parts.append(Fp.mean(0))
            if "slope" in kind:
                t = np.linspace(0, 1, n)
                tc = t - t.mean()
                parts.append(((Fp - Fp.mean(0)) * tc[:, None]).sum(0) / ((tc ** 2).sum() or 1.0))
            if "contrast" in kind:
                k = max(n // 3, 1)
                parts.append(Fp[-k:].mean(0) - Fp[:k].mean(0))
            if "figo" in kind:
                parts.append(np.array([(g == 2).mean(), (g >= 1).mean(), float(g.max()),
                                       (g[-max(n // 3, 1):] == 2).mean()]))
            rows.append(np.concatenate(parts))
        return np.array(rows)

    base = []
    for m, teP in grouped_folds(D):
        s = lr().fit(D["F"][m], D["y"][m]).predict_proba(D["F"])[:, 1]
        pl = np.array([int(D["y"][D["pidx"][p]].max()) for p in teP])
        base.append(roc_auc_score(pl, np.array([s[D["pidx"][p]].max() for p in teP])))
    out("| representation | n feats | patient AUROC |")
    out("|---|---:|---:|")
    out(f"| per-window model, max-aggregated (baseline) | 19 | "
        f"{np.mean(base):.4f} +/- {np.std(base):.4f} |")
    for kind in ("max", "max+mean", "max+mean+slope", "max+mean+slope+contrast",
                 "max+mean+slope+contrast+figo", "slope", "contrast", "figo"):
        A = build(kind)
        aucs = []
        for rep in range(N_REPEATS):
            for tr, te in StratifiedKFold(N_FOLDS, shuffle=True,
                                          random_state=rep).split(A, D["plab"]):
                c = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000, C=0.1,
                                                     class_weight="balanced"))
                c.fit(A[tr], D["plab"][tr])
                aucs.append(roc_auc_score(D["plab"][te], c.predict_proba(A[te])[:, 1]))
        out(f"| {kind} | {A.shape[1]} | {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} |")
    out("\nEvery patient-level representation loses to the per-window model plus "
        "aggregation. One clean representation per patient means 547 training rows "
        "instead of 8,517; the better-shaped question does not pay for the data.\n")


def probe_clustering(D, out):
    out("## clustering -- what an independence assumption costs\n")
    oof = np.zeros(len(D["y"]))
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True,
                                  random_state=0).split(D["patients"], D["plab"]):
        trS = set(D["patients"][tr])
        m = np.array([q in trS for q in D["pid"]])
        oof[~m] = lr().fit(D["F"][m], D["y"][m]).predict_proba(D["F"][~m])[:, 1]
    auc = roc_auc_score(D["y"], oof)
    rng = np.random.default_rng(0)

    def boot(by_patient, n=2000):
        vals = []
        for _ in range(n):
            if by_patient:
                ps = rng.choice(D["patients"], len(D["patients"]), replace=True)
                b = np.concatenate([D["pidx"][p] for p in ps])
            else:
                b = rng.choice(len(D["y"]), len(D["y"]), replace=True)
            if len(np.unique(D["y"][b])) < 2:
                continue
            vals.append(roc_auc_score(D["y"][b], oof[b]))
        return np.percentile(vals, [2.5, 97.5])

    lo_w, hi_w = boot(False)
    lo_p, hi_p = boot(True)
    out(f"Out-of-fold window AUROC (patient-grouped): {auc:.4f}\n")
    out("| bootstrap unit | 95% CI | width |")
    out("|---|---|---:|")
    out(f"| windows (assumes independence) | [{lo_w:.4f}, {hi_w:.4f}] | {hi_w - lo_w:.4f} |")
    out(f"| patients (respects clustering) | [{lo_p:.4f}, {hi_p:.4f}] | {hi_p - lo_p:.4f} |")
    out(f"\nTreating overlapping windows as independent understates the interval by "
        f"{(hi_p - lo_p) / (hi_w - lo_w):.1f}x. Any p-value computed under that "
        f"assumption -- DeLong's included -- is overstated by a corresponding amount.\n")


PROBE_FNS = {"protocol": probe_protocol, "identity": probe_identity,
             "label": probe_label, "aggregation": probe_aggregation,
             "target": probe_target, "cohort": probe_cohort,
             "representation": probe_representation, "clustering": probe_clustering}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/processed_clinical/")
    ap.add_argument("--probes", default=",".join(PROBES),
                    help=f"comma-separated subset of: {', '.join(PROBES)}")
    ap.add_argument("--out", default="docs/evaluation_protocol_evidence.md")
    args = ap.parse_args()

    chosen = [p.strip() for p in args.probes.split(",") if p.strip()]
    bad = [p for p in chosen if p not in PROBE_FNS]
    if bad:
        sys.exit(f"[ABORT] unknown probe(s): {bad}. choose from {PROBES}")

    D = load_all(args.data_dir)
    D["_dir"] = args.data_dir

    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("# Evaluation-protocol evidence\n")
    out("Generated by `scripts/audit_evaluation_protocol.py`. No network, no GPU -- "
        "reproducible offline from `data/processed_clinical/`.\n")
    out(f"Cohort: {len(D['y'])} windows | {len(D['patients'])} patients | "
        f"{int(D['plab'].sum())} positive ({100 * D['plab'].mean():.1f}%) | "
        f"{len(D['y']) / len(D['patients']):.1f} windows per patient\n")
    out("> The `protocol` probe is a negative control describing what this benchmark "
        "permits. It is NOT an accusation about any published paper -- Dang et al. "
        "report patient-grouped folds and the 2026 foundation model splits at "
        "recording level. See the module docstring.\n")

    for name in chosen:
        out("\n---\n")
        PROBE_FNS[name](D, out)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
