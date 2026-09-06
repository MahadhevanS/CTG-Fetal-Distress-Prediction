"""Is pH<=7.15 the most CTG-predictable outcome available? Same features, same
patient partition, different targets. BDecf isolates METABOLIC acidosis, which is
the physiology CTG actually reflects; pH mixes respiratory + metabolic."""
import sys, os, gc, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.training.protocol import Protocol
D  = "data/processed_clinical"
SC = "results/phase8"
F,PID,PH,BD,AP,YP,H = [],[],[],[],[],[],[]
for sp in ("train","val","test"):
    d = torch.load(os.path.join(D,f"{sp}_dataset.pt"), weights_only=False)
    F.append(np.hstack([d["y_features"].numpy(), np.load(os.path.join(D,f"{sp}_extended_features.npy"))]))
    PID.append(np.array([m[0] for m in d["metadata"]]))
    PH.append(d["y_ph"].numpy()); BD.append(d["y_bdecf"].numpy()); AP.append(d["y_apgar5"].numpy())
    YP.append(d["y_primary"].numpy()); H.append(np.load(os.path.join(SC,f"hrv_{sp}.npy")))
    del d; gc.collect()
F19=np.concatenate(F); pid=np.concatenate(PID)
ph=np.concatenate(PH); bd=np.concatenate(BD); ap=np.concatenate(AP); yp=np.concatenate(YP)
H=np.nan_to_num(np.concatenate(H),nan=0.,posinf=0.,neginf=0.)

def evaluate(name, yv):
    ok = np.isfinite(yv)
    if ok.mean() < 1.0:
        # keep every window; impute label at patient level is unsafe -> drop those patients
        keep_p = {p for p in set(pid) if np.isfinite(yv[pid==p]).all()}
        ok = np.array([p in keep_p for p in pid])
    y = yv[ok].astype(int); pp = pid[ok]; M = F19[ok]
    import json
    blob = json.load(open(os.path.join(D,"folds.json")))
    asg = {k:v for k,v in blob["assignment"].items() if k in set(pp)}
    P = Protocol(pp, y, asg)
    oof=np.zeros(len(M)); cnt=np.zeros(len(M))
    for tr, te_p in P.folds(repeat=0):
        te=np.isin(pp,te_p)
        clf=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                          LogisticRegression(max_iter=5000,C=0.1,class_weight="balanced"))
        clf.fit(M[tr],y[tr]); oof[te]+=clf.predict_proba(M[te])[:,1]; cnt[te]+=1
    return P.report(name, oof/np.maximum(cnt,1), how="max")

print("--- same 19 features, same patient folds, DIFFERENT TARGETS ---")
evaluate("pH <= 7.15            (current target)", (ph<=7.15).astype(float))
evaluate("pH <= 7.05            (severe)", (ph<=7.05).astype(float))
evaluate("BDecf >= 12  (metabolic acidaemia)", np.where(np.isfinite(bd),(bd>=12).astype(float),np.nan))
evaluate("BDecf >= 10", np.where(np.isfinite(bd),(bd>=10).astype(float),np.nan))
evaluate("BDecf >= 8", np.where(np.isfinite(bd),(bd>=8).astype(float),np.nan))
evaluate("Apgar5 < 7", np.where(np.isfinite(ap),(ap<7).astype(float),np.nan))
evaluate("pH<=7.15 AND BDecf>=8 (concordant)",
         np.where(np.isfinite(bd),((ph<=7.15)&(bd>=8)).astype(float),np.nan))
for nm,v in [("pH",ph),("BDecf",bd),("Apgar5",ap)]:
    pv={}
    for p in set(pid): pv[p]=v[pid==p][0]
    a=np.array(list(pv.values())); print(f"  {nm}: n={np.isfinite(a).sum()} of {len(a)} patients")
