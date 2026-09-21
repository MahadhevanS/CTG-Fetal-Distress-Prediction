import sys, os, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, gc
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from src.training.protocol import Protocol
D  = "data/processed_clinical"
SC = "results/phase8"
F,Y,PID,SEC,H = [],[],[],[],[]
for sp in ("train","val","test"):
    d = torch.load(os.path.join(D,f"{sp}_dataset.pt"), weights_only=False)
    F.append(np.hstack([d["y_features"].numpy(), np.load(os.path.join(D,f"{sp}_extended_features.npy"))]))
    Y.append(d["y_primary"].numpy()); PID.append(np.array([m[0] for m in d["metadata"]]))
    SEC.append(d["w_is_second_stage"].numpy()); H.append(np.load(os.path.join(SC,f"hrv_{sp}.npy")))
    del d; gc.collect()
F19=np.concatenate(F); y=np.concatenate(Y); pid=np.concatenate(PID)
sec=np.concatenate(SEC); H=np.concatenate(H)
H=np.nan_to_num(H, nan=0., posinf=0., neginf=0.)
print(f"windows {len(y)}  patients {len(set(pid))}")
P = Protocol.load_or_create(pid, y)
def run(name, M):
    oof=np.zeros(len(M)); cnt=np.zeros(len(M))
    for tr, te_p in P.folds(repeat=0):
        te=np.isin(pid,te_p)
        clf=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                          LogisticRegression(max_iter=5000,C=0.1,class_weight="balanced"))
        clf.fit(M[tr],y[tr]); oof[te]+=clf.predict_proba(M[te])[:,1]; cnt[te]+=1
    return P.report(name, oof/np.maximum(cnt,1), how="max")
print("\n--- patient-level AUROC, max agg, frozen protocol, repeat 0 ---")
r={}
r["19"]      = run("19 clinical descriptors (baseline)", F19)
r["hrv"]     = run("18 HRV / nonlinear only", H)
r["19+hrv"]  = run("19 + 18 HRV", np.hstack([F19,H]))
r["19+hrv+s"]= run("19 + HRV + stage-II flag", np.hstack([F19,H,sec[:,None]]))
b=r["19"]["auroc"]
print(f"\nbaseline {b:.4f} | pre-registered bar = MDE 0.0642")
for k,v in r.items():
    if k!="19": print(f"  {k:10s} {v['auroc']-b:+.4f}")
# which HRV features carry any univariate signal at patient level?
lab,_=P.to_patient(np.zeros(len(y)))
print("\nunivariate patient-level AUROC of each HRV feature (max-agg):")
NAMES=["vlf","lf","mf","hf","lf_hf","mf_tot","logtot","sd1","sd2","sd_ratio",
       "sampen","dfa_a1","dfa_a2","stv_dawes","ltv_minmax","skew","kurt","zerocross"]
sc=[]
for j,n in enumerate(NAMES):
    _,s=P.to_patient(H[:,j],how="max"); a=roc_auc_score(lab,s)
    sc.append((max(a,1-a),n))
for a,n in sorted(sc,reverse=True)[:8]: print(f"  {n:10s} {a:.4f}")
json.dump({k:{kk:(float(vv) if isinstance(vv,(int,float,np.floating)) else vv) for kk,vv in v.items()} for k,v in r.items()},
          open(os.path.join(SC,"hrv_res.json"),"w"), indent=1)
