"""
Archive a trained model with full provenance so it can never become another
"0.8565" -- a number we could not reproduce because the weights, the exact data
that produced them, and the code state were not captured together.

Captures, in one directory:
  - the checkpoint weights (copied, not referenced)
  - SHA256 of every data file used (detects silent dataset regeneration, the
    exact failure mode that killed the 0.8565 result)
  - git commit SHA + dirty-file list
  - the exact command, config, and reported metrics
  - environment (python/torch/cuda versions)

Usage:
  python scripts/archive_model.py --name standalone_mil_foldmatched \
      --checkpoint_dir checkpoints/ctg_crossformer_mil_foldmatched/ \
      --data_dir data/processed_mil/ \
      --command "python src/models/train_ctg_crossformer.py --data_dir data/processed_mil/ --fold_mode patient_level" \
      --metrics '{"cv_auroc_mean": 0.865}'
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, platform
from datetime import datetime

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--command", default="")
    ap.add_argument("--config", default=None)
    ap.add_argument("--metrics", default="{}")
    ap.add_argument("--notes", default="")
    ap.add_argument("--archive_root", default="archive/")
    a = ap.parse_args()

    dest = os.path.join(a.archive_root, a.name)
    if os.path.exists(dest):
        sys.exit(f"[ABORT] {dest} already exists -- refusing to overwrite an archived model.")
    os.makedirs(os.path.join(dest, "checkpoints"), exist_ok=True)

    ckpts = {}
    for f in sorted(os.listdir(a.checkpoint_dir)):
        if f.endswith(".pth"):
            src = os.path.join(a.checkpoint_dir, f)
            shutil.copy2(src, os.path.join(dest, "checkpoints", f))
            ckpts[f] = {"sha256": sha256(src), "bytes": os.path.getsize(src)}

    data_files = {}
    for f in sorted(os.listdir(a.data_dir)):
        p = os.path.join(a.data_dir, f)
        if os.path.isfile(p):
            data_files[f] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}

    if a.config and os.path.exists(a.config):
        shutil.copy2(a.config, os.path.join(dest, os.path.basename(a.config)))

    try:
        import torch
        torch_v, cuda_v = torch.__version__, torch.version.cuda
    except Exception:
        torch_v = cuda_v = None

    manifest = {
        "name": a.name,
        "archived_utc": datetime.utcnow().isoformat() + "Z",
        "command": a.command,
        "config": a.config,
        "metrics": json.loads(a.metrics),
        "notes": a.notes,
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": (git("status", "--porcelain") or "").splitlines(),
        },
        "data": {"dir": a.data_dir, "files": data_files},
        "checkpoints": ckpts,
        "env": {"python": platform.python_version(), "torch": torch_v, "cuda": cuda_v,
                "platform": platform.platform()},
    }
    with open(os.path.join(dest, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] archived -> {dest}")
    print(f"     {len(ckpts)} checkpoints, {len(data_files)} data files hashed")
    print(f"     git commit: {manifest['git']['commit']}")
    if manifest["git"]["dirty_files"]:
        print(f"     WARNING: {len(manifest['git']['dirty_files'])} uncommitted files at archive time")

if __name__ == "__main__":
    main()
