import numpy as np, glob, os, re
RAW = "data/raw/ctu-chb-intrapartum"
recs = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(RAW,"*.hea")))
rows=[]
for rec in recs:
    a=np.fromfile(os.path.join(RAW,rec+".dat"),dtype="<i2").reshape(-1,2)
    uc=a[:,1]/100.0; fhr=a[:,0]/100.0
    seg=uc[-14400:]                      # the 60-min crop actually used
    miss=float((seg<=0).mean())
    v=seg[seg>0]
    if len(v)<1200: rows.append((rec,miss,np.nan,np.nan,np.nan)); continue
    # is there real contraction structure? count peaks with prominence>10 units
    from scipy.signal import find_peaks
    pk,pr=find_peaks(v,prominence=10,distance=int(4*60))
    rate=len(pk)/(len(v)/4/60)*10        # contractions per 10 min
    rows.append((rec,miss,float(v.std()),rate,float(np.ptp(v))))
import numpy as np
miss=np.array([r[1] for r in rows]); sd=np.array([r[2] for r in rows],float)
rate=np.array([r[3] for r in rows],float); rng=np.array([r[4] for r in rows],float)
print("UC in the 60-min crop, n=%d"%len(rows))
print("  missing/zero fraction   median=%.3f  >50%% missing: %d records"%(np.nanmedian(miss),(miss>0.5).sum()))
print("  UC sd                   median=%.2f  <5 (near-flat): %d"%(np.nanmedian(sd),(sd<5).sum()))
print("  detected contractions/10min median=%.2f"%np.nanmedian(rate))
print("  implausible rate (<1 or >8 per 10min): %d records (%.1f%%)"%(((rate<1)|(rate>8)).sum(),100*((rate<1)|(rate>8)).mean()))
usable=(miss<0.5)&(sd>=5)&(rate>=1)&(rate<=8)
print("  USABLE UC (all criteria): %d/%d = %.1f%%"%(usable.sum(),len(rows),100*usable.mean()))
np.save(r"results/phase8/uc_usable.npy",
        np.array([[int(r[0]) for r in rows],usable.astype(int)]))
