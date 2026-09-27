"""
Standalone scorer for the frozen G-A model (results/phase22_ga_final/frozen_model/).  Exploratory model: NOT the locked URM.

    frozen = load("results/phase22_ga_final/frozen_model")
    out = score_patient(frozen, X40_windows, elapsed_min, parity)
      X40_windows : (T, 40) locked 40-D state-trajectory vectors, chronological (one per 20-min window, 2.5-min stride)
      elapsed_min : (T,)   minutes since the patient's first retained window
      parity      : maternal parity (number of previous births)
    returns {"window_score": (T,), "pooled_score": (T,), "risk": (T,)}   -- `risk[t]` is the causal risk after window t
"""
import os, sys, json
import numpy as np
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from src.models.phase16_causal_attention import AttentionScorer, predict_all_prefixes


def load(model_dir):
    m = json.load(open(os.path.join(model_dir, "model.json")))
    scorers = []
    for s in m["pooler_seeds"]:
        sc = AttentionScorer(m["pooler_in_dim"], hidden=m["pooler_hidden"]); sc.load_state_dict(torch.load(os.path.join(model_dir, f"pooler_seed{s}.pt"), weights_only=True)); sc.eval(); scorers.append(sc)
    return {"cfg": m, "scorers": scorers}


def score_patient(frozen, X40, elapsed_min, parity):
    m = frozen["cfg"]; w = m["window_model"]
    A = (np.asarray(X40, float) - np.array(w["feature_mean"])) / np.array(w["feature_scale"])
    raw = A @ np.array(w["ridge_coef"]) + w["ridge_intercept"]
    r = 1.0 / (1.0 + np.exp(-(w["platt_coef"] * (raw - w["platt_mu"]) / w["platt_sd"] + w["platt_intercept"])))
    rt = torch.tensor(r, dtype=torch.float32); et = torch.tensor(np.asarray(elapsed_min), dtype=torch.float32)
    z = np.mean([predict_all_prefixes(s, rt, et, True, len(r)) for s in frozen["scorers"]], axis=0)
    p = m["parity"]; lg = p["coef"] * (parity - p["mean"]) / p["scale"] + p["intercept"]; par = 1.0 / (1.0 + np.exp(-lg))
    a = m["fusion_alpha"]
    return {"window_score": r, "pooled_score": z, "risk": a * z + (1 - a) * par}
