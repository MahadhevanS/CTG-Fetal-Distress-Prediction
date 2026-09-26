"""
Phase 20 Stage 0 supplement to G4: A/A test. Baseline B0 with network seeds {42,43,44} vs the identical pipeline
with seeds {45,46,47}. Nothing about the model differs, so any nonzero dM is pure training noise: this is the null
distribution the Stage 1 ADOPT bar (dM >= +0.005) has to be read against.
"""
import os, sys, json
import numpy as np, pandas as pd
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
import scripts.phase20_harness as hz

ALT = [45, 46, 47]

def main():
    c = hz.load_ctu = hz.load_ctx()
    y = np.array([c.y[p] for p in c.d464]); tdel = hz.tdel_by_pid(c, c.d464)
    base = np.load(os.path.join(hz.OUT_DIR, "stage0_baseline_D464_scores.npz"))
    rows = []
    for name, assign in hz.split_assignments(c, c.d464).items():
        off = 0 if name == "CANONICAL" else 100 + int(name.split("_")[1])
        pw = hz.oof_window_scores(c, c.X40, c.d464, assign)
        seq, _ = hz.pooler_oof_sequences(c, pw, c.d464, assign, seeds=ALT, offset=off)
        r, _ = hz.urm_cv(c, c.d464, assign, seq, tdel)
        sc = {"B0": base[name], "B0_alt": r["URM"]}
        boot = hz.boot_all(y, sc)
        rec = {"split": name, **hz.summarize_delta(boot, "B0_alt", "B0", y, sc)}
        rows.append(rec); print("  A/A", {k: round(v, 4) for k, v in rec.items() if k != "split"} | {"split": name}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(hz.OUT_DIR, "stage0_AA_test.csv"), index=False)
    print(f"  A/A dM: mean {df.dM.mean():+.4f}  SD {df.dM.std(ddof=1):.4f}  max|dM| {df.dM.abs().max():.4f}  "
          f"min per-horizon d {min(df[[f'd{h}' for h in hz.H]].min()):+.4f}  n(|dM|>=0.005)={(df.dM.abs()>=0.005).sum()}/6")

if __name__ == "__main__":
    main()
