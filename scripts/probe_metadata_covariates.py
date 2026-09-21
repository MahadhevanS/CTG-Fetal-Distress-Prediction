"""Patient-level: do metadata covariates add over the 19 CTG descriptors?
Separates PROSPECTIVE (known during labour) from RETROSPECTIVE (known only at delivery)."""
import sys, os, re, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.training.protocol import Protocol

D = "data/processed_clinical"
RAW = "data/raw/ctu-chb-intrapartum"

F, Y, PID, SEC, END = [], [], [], [], []
for sp in ("train","val","test"):
    d = torch.load(os.path.join(D,f"{sp}_dataset.pt"), weights_only=False)
    ext = np.load(os.path.join(D,f"{sp}_extended_features.npy"))
    F.append(np.hstack([d["y_features"].numpy(), ext]))
    Y.append(d["y_primary"].numpy()); PID.append(np.array([m[0] for m in d["metadata"]]))
    SEC.append(d["w_is_second_stage"].numpy()); END.append(d["w_minutes_before_end"].numpy())
    del d
F19=np.concatenate(F); y=np.concatenate(Y); pid=np.concatenate(PID)
sec=np.concatenate(SEC); end=np.concatenate(END)

def hdr(rec):
    o={}
    for line in open(os.path.join(RAW,rec+".hea")):
        m=re.match(r"#(.+?)\s\s+(\S+)\s*$",line.strip())
        if m: o[m.group(1).strip()]=m.group(2)
    return o
def num(v):
    try: return float(v)
    except Exception: return np.nan

PROSPECTIVE = ["Age","Gravidity","Parity","Gest. weeks","Diabetes","Hypertension",
               "Preeclampsia","Pyrexia","Meconium","Liq. praecox","Induced",
               "Presentation","I.stage","Rec. type"]
RETROSPECTIVE = ["II.stage","Deliv. type","NoProgress","CK/KP","Weight(g)","Sex"]

H={r:hdr(r) for r in sorted(set(pid))}
def block(names):
    return np.array([[num(H[r].get(n,"nan")) for n in names] for r in pid], float)
Mp, Mr = block(PROSPECTIVE), block(RETROSPECTIVE)

P = Protocol.load_or_create(pid, y)
def run(name, M):
    oof=np.zeros(len(M)); cnt=np.zeros(len(M))
    for tr, te_p in P.folds(repeat=0):
        te=np.isin(pid,te_p)
        clf=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                          LogisticRegression(max_iter=3000,C=0.1,class_weight="balanced"))
        clf.fit(M[tr],y[tr]); oof[te]+=clf.predict_proba(M[te])[:,1]; cnt[te]+=1
    return P.report(name, oof/np.maximum(cnt,1), how="max")

print("\n--- patient-level AUROC, max agg, frozen protocol, repeat 0 ---")
b=run("19 CTG descriptors (baseline)", F19)["auroc"]
r={}
r["prosp only"]  = run("14 PROSPECTIVE covariates only", Mp)
r["19+prosp"]    = run("19 + 14 prospective", np.hstack([F19,Mp]))
r["19+time"]     = run("19 + stage-II flag + min-before-end", np.hstack([F19,sec[:,None],end[:,None]]))
r["19+pr+time"]  = run("19 + prospective + time", np.hstack([F19,Mp,sec[:,None],end[:,None]]))
r["retro only"]  = run("6 RETROSPECTIVE (leaky) only", Mr)
r["19+retro"]    = run("19 + retrospective (LEAKY, ref only)", np.hstack([F19,Mr]))
print(f"\nbaseline {b:.4f} | MDE 0.0642")
for k,v in r.items(): print(f"  {k:14s} {v['auroc']-b:+.4f}")
