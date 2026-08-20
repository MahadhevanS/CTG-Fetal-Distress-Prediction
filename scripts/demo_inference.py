"""End-to-end demo: raw CTG recording in, structured clinical output out."""
import json, os, sys
import numpy as np
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "src", "preprocessing"))
from ingestion import load_ctu_chb_record, load_clinical_metadata
from src.inference.ctg_inference import CTGInference

RECORD = sys.argv[1] if len(sys.argv) > 1 else "2045"

md = load_clinical_metadata("data/raw/ctu-chb-intrapartum/clinical_metadata.csv")
md.columns = [c.strip().lower() for c in md.columns]
ph = float(md.set_index("record_id").loc[int(RECORD), "ph"])

print("=" * 74)
print(" INPUT")
print("=" * 74)
fhr, uc, fs = load_ctu_chb_record(os.path.join("data/raw/ctu-chb-intrapartum", RECORD))
print(f"  raw recording      : record {RECORD}")
print(f"  FHR                : {fhr.shape[0]} samples @ {fs:g} Hz  = {len(fhr)/(fs*60):.1f} min")
print(f"  UC                 : {uc.shape[0]} samples")
print(f"  FHR range          : {fhr[fhr>0].min():.0f} - {fhr.max():.0f} bpm")
print(f"  missing samples    : {np.mean(fhr==0)*100:.1f}%")
print(f"  (ground truth, not given to the model: umbilical pH {ph:.3f}"
      f" -> {'ACIDOTIC' if ph <= 7.15 else 'normal'})")

eng = CTGInference("checkpoints/ctg_crossformer_crp",
                   "data/processed_mil/ctu_signal_scaler.npz", threshold=0.30)
# last 60 min, as in training
LAST = int(60 * 60 * fs)
out = eng.analyse(fhr[-LAST:], uc[-LAST:], patient_id=RECORD)

print()
print("=" * 74); print(" OUTPUT (structured, as the device receives it)"); print("=" * 74)
compact = dict(out); compact["windows"] = out["windows"][:2] + ["... truncated ..."]
print(json.dumps(compact, indent=2)[:2100])
print()
print("=" * 74); print(" SUMMARY"); print("=" * 74)
s = out["summary"]
print(f"  peak risk {s['peak_risk']:.3f} at minute {s['peak_at_min']:.0f} | "
      f"mean {s['mean_risk']:.3f} | {s['n_flagged']}/{out['windows_analysed']} flagged")
print(f"  trend {s['risk_trend_per_hour']:+.3f}/hour | longest run {s['longest_sustained_windows']} windows")
print(f"  recurring findings: {s['recurring_findings'] or 'none'}")
