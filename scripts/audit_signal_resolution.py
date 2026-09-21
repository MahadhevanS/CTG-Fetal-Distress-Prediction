import numpy as np, glob, os, re
from collections import defaultdict

RAW = "data/raw/ctu-chb-intrapartum"

def hdr(rec):
    d = {}
    for line in open(os.path.join(RAW, rec + ".hea")):
        m = re.match(r"#(.+?)\s\s+(\S+)\s*$", line.strip())
        if m: d[m.group(1).strip()] = m.group(2)
    return d

def load(rec):
    a = np.fromfile(os.path.join(RAW, rec + ".dat"), dtype="<i2").reshape(-1, 2)
    return a[:, 0] / 100.0, a[:, 1] / 100.0

recs = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(RAW, "*.hea")))
print("records:", len(recs))

stats = defaultdict(list)
for rec in recs:
    h = hdr(rec)
    rt = int(float(h.get("Rec. type", -1)))
    rt = 2 if rt in (2, 12) else rt
    fhr, uc = load(rec)
    valid = fhr > 0
    f = fhr[valid]
    if len(f) < 4000: continue
    # quantization: smallest nonzero absolute step between consecutive valid samples
    d = np.abs(np.diff(f))
    d = d[d > 0]
    q = np.min(d) if len(d) else np.nan
    # high-frequency power fraction (>0.5 Hz) on a long contiguous valid run
    runs, s = [], None
    for i, v in enumerate(valid):
        if v and s is None: s = i
        elif not v and s is not None:
            runs.append((s, i)); s = None
    if s is not None: runs.append((s, len(valid)))
    runs.sort(key=lambda r: r[1]-r[0], reverse=True)
    hf = np.nan
    if runs and runs[0][1]-runs[0][0] >= 2400:
        seg = fhr[runs[0][0]:runs[0][1]]
        seg = seg - seg.mean()
        P = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))**2
        fr = np.fft.rfftfreq(len(seg), d=0.25)
        tot = P[(fr > 0.01)].sum()
        hf = P[(fr > 0.5)].sum() / tot if tot > 0 else np.nan
    stats[rt].append((q, hf, 1 - valid.mean(), np.std(np.diff(f))))

for rt in sorted(stats):
    a = np.array(stats[rt], dtype=float)
    lab = {1: "Doppler US", 2: "scalp FECG", -1: "unknown"}.get(rt, rt)
    print(f"\n{lab:12s} n={len(a)}")
    print(f"  min quantization step (bpm)  median={np.nanmedian(a[:,0]):.4f}")
    print(f"  HF power fraction >0.5Hz     median={np.nanmedian(a[:,1]):.4f}")
    print(f"  missing fraction             median={np.nanmedian(a[:,2]):.4f}")
    print(f"  sd of sample-to-sample diff  median={np.nanmedian(a[:,3]):.4f}")
