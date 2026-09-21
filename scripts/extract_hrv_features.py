"""Extract HRV/nonlinear features one split at a time (memory-bounded)."""
import sys, os, gc, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from scipy import signal as sps
D  = "data/processed_clinical"
SC = "results/phase8"
FS = 4.0
HRV_NAMES = ["vlf","lf","mf","hf","lf_hf","mf_tot","logtot","sd1","sd2","sd_ratio",
             "sampen","dfa_a1","dfa_a2","stv_dawes","ltv_minmax","skew","kurt","zerocross"]

def dfa_alpha(x, lo, hi):
    scales = np.unique(np.logspace(np.log10(lo), np.log10(hi), 8).astype(int))
    scales = scales[scales >= 4]
    yy = np.cumsum(x - x.mean()); F = []
    for s in scales:
        n = len(yy)//s
        if n < 2: F.append(np.nan); continue
        seg = yy[:n*s].reshape(n, s).T
        A = np.vstack([np.arange(s), np.ones(s)]).T
        coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
        F.append(np.sqrt(((seg - A@coef)**2).mean()))
    F = np.array(F); ok = np.isfinite(F) & (F > 0)
    return float(np.polyfit(np.log(scales[ok]), np.log(F[ok]), 1)[0]) if ok.sum() >= 3 else np.nan

def sampen(x, m=2, r=0.2):
    x = np.asarray(x[:400], dtype=np.float32)
    if len(x) < 100: return np.nan
    sd = x.std()
    if sd == 0: return np.nan
    tol = np.float32(r*sd)
    def cnt(mm):
        N = len(x)-mm+1
        e = np.lib.stride_tricks.sliding_window_view(x, mm)[:N]
        c = 0
        for k in range(N):                       # row-wise: no N*N*mm temporary
            c += int((np.abs(e[k]-e).max(1) <= tol).sum())
        return c - N                             # drop self-matches
    A, B = cnt(m+1), cnt(m)
    return float(-np.log(A/B)) if A > 0 and B > 0 else np.nan

def hrv(fhr_bc, mask):
    out = np.full(len(HRV_NAMES), np.nan, np.float32)
    valid = mask < 0.5
    if valid.sum() < FS*120: return out
    x = fhr_bc.astype(np.float64).copy(); idx = np.arange(len(x))
    if (~valid).any(): x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    x -= x.mean()
    fr, P = sps.welch(x, fs=FS, nperseg=min(1024, len(x)))
    bp = lambda a,b: (float(np.trapezoid(P[(fr>=a)&(fr<b)], fr[(fr>=a)&(fr<b)]))
                      if ((fr>=a)&(fr<b)).sum() > 1 else 0.0)
    vlf,lf,mf,hf = bp(.0033,.04), bp(.04,.15), bp(.15,.5), bp(.5,1.0)
    tot = vlf+lf+mf+hf+1e-12
    out[0:4] = [vlf,lf,mf,hf]; out[4] = lf/(hf+1e-12); out[5] = mf/tot; out[6] = np.log10(tot)
    d1 = np.diff(x)
    sd1 = np.sqrt(0.5)*d1.std(); sd2 = np.sqrt(max(2*x.var()-0.5*d1.var(), 1e-12))
    out[7:10] = [sd1, sd2, sd1/(sd2+1e-12)]
    dec = x[::4]
    out[10] = sampen(dec); out[11] = dfa_alpha(dec,4,16); out[12] = dfa_alpha(dec,16,64)
    ep = int(3.75*FS); n = len(x)//ep
    em = x[:n*ep].reshape(n,ep).mean(1)
    out[13] = float(np.abs(np.diff(em)).mean()); out[14] = float(em.max()-em.min())
    s = x.std()+1e-12
    out[15] = float(((x/s)**3).mean()); out[16] = float(((x/s)**4).mean())
    out[17] = float((np.diff(np.sign(x)) != 0).mean())
    return out

for sp in ("test","val","train"):
    dst = os.path.join(SC, f"hrv_{sp}.npy")
    if os.path.exists(dst): print(f"{sp}: cached", flush=True); continue
    d = torch.load(os.path.join(D, f"{sp}_dataset.pt"), weights_only=False)
    X = d["X"].numpy(); del d; gc.collect()
    H = np.zeros((len(X), len(HRV_NAMES)), np.float32); t0 = time.time()
    for i in range(len(X)):
        H[i] = hrv(X[i,0], X[i,2])
        if i and i % 500 == 0:
            print(f"  {sp} {i}/{len(X)}  {(time.time()-t0)/i*1000:.0f} ms/win", flush=True)
    np.save(dst, H); del X; gc.collect()
    print(f"{sp}: {H.shape} done in {time.time()-t0:.0f}s  nan={np.isnan(H).mean():.3f}", flush=True)
print("EXTRACTION COMPLETE", flush=True)
