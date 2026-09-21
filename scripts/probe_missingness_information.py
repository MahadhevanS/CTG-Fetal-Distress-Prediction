"""Is the MISSINGNESS PATTERN itself predictive, beyond the signal it hides?"""
import sys, os, gc, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, json
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from src.training.protocol import Protocol
D  = "data/processed_clinical"
def gapfeats(m):                       # m = missingness mask, 1 = missing
    f = np.zeros(6, np.float32)
    f[0] = m.mean()
    d = np.diff(np.concatenate(([0], (m > .5).astype(int), [0])))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    L = en - st
    f[1] = len(L); f[2] = L.max()/4 if len(L) else 0; f[3] = L.mean()/4 if len(L) else 0
    h = len(m)//2
    f[4] = m[h:].mean() - m[:h].mean()          # is loss worsening within window?
    f[5] = float((m[-4*60:] > .5).mean())       # loss in the final minute
    return f
F,Y,PID,Q,G = [],[],[],[],[]
for sp in ("train","val","test"):
    d = torch.load(os.path.join(D,f"{sp}_dataset.pt"), weights_only=False)
    X = d["X"].numpy()
    G.append(np.array([gapfeats(X[i,2]) for i in range(len(X))]))
    F.append(np.hstack([d["y_features"].numpy(), np.load(os.path.join(D,f"{sp}_extended_features.npy"))]))
    Y.append(d["y_primary"].numpy()); PID.append(np.array([m[0] for m in d["metadata"]]))
    Q.append(d["w_quality"].numpy()); del d, X; gc.collect()
F19=np.concatenate(F); y=np.concatenate(Y); pid=np.concatenate(PID)
q=np.concatenate(Q); G=np.concatenate(G)
P = Protocol.load_or_create(pid, y)
def run(name, M):
    oof=np.zeros(len(M)); cnt=np.zeros(len(M))
    for tr, te_p in P.folds(repeat=0):
        te=np.isin(pid,te_p)
        clf=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                          LogisticRegression(max_iter=5000,C=0.1,class_weight="balanced"))
        clf.fit(M[tr],y[tr]); oof[te]+=clf.predict_proba(M[te])[:,1]; cnt[te]+=1
    return P.report(name, oof/np.maximum(cnt,1), how="max")
print("--- missingness pattern, frozen protocol ---")
b=run("19 clinical descriptors (baseline)", F19)["auroc"]
r={}
r["gaps only"] = run("6 missingness-pattern features only", G)
r["19+gaps"]   = run("19 + missingness pattern", np.hstack([F19,G]))
print(f"\nbaseline {b:.4f} | bar 0.0642")
for k,v in r.items(): print(f"  {k:10s} {v['auroc']-b:+.4f}")
lab,_=P.to_patient(np.zeros(len(y)))
print("\nunivariate patient-level AUROC:")
for j,n in enumerate(["miss_frac","n_gaps","max_gap_s","mean_gap_s","loss_trend","final_min_loss"]):
    for how in ("max","mean"):
        _,s=P.to_patient(G[:,j],how=how); a=roc_auc_score(lab,s)
        print(f"  {n:14s} ({how:4s}) {a:.4f}")
