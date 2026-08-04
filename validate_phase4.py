"""Quick validation script for all Phase 4+ modules."""
import numpy as np
import torch
import torch.nn as nn

# Test calibration
from src.training.calibration import ThresholdOptimizer, TemperatureScaler
np.random.seed(42)
y_true = np.random.randint(0, 2, 200)
y_probs = np.clip(np.random.randn(200) * 0.3 + 0.5, 0, 1)
opt = ThresholdOptimizer(objective="f1")
thresh, score = opt.fit(y_probs, y_true)
print(f"ThresholdOptimizer: best_thresh={thresh:.3f}, score={score:.4f}")

ts = TemperatureScaler()
logits = np.log(y_probs / (1 - y_probs + 1e-8) + 1e-8)
T = ts.fit(logits, y_true)
cal_probs = ts.calibrate(logits)
ece = ts.expected_calibration_error(cal_probs, y_true)
print(f"TemperatureScaler: T={T:.3f}, ECE={ece:.4f}")

# Test SWA
from src.training.checkpoint_utils import ExponentialMovingAverage, StochasticWeightAveraging
swa = StochasticWeightAveraging(max_checkpoints=3)
m = nn.Linear(10, 1)
for _ in range(5):
    swa.add_checkpoint(m.state_dict())
avg = swa.average()
print(f"SWA: averaged {swa.n_checkpoints} checkpoints, keys={list(avg.keys())}")

# Test EMA
ema = ExponentialMovingAverage(m, decay=0.999)
ema.update()
with ema.average_parameters():
    pass
print("EMA: context manager works [OK]")

# Test Sampler
from src.training.samplers import BalancedBatchSampler, HardExampleMiner
labels = torch.tensor([1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0])
sampler = BalancedBatchSampler(labels, batch_size=4)
batches = list(sampler)
print(f"BalancedBatchSampler: {len(batches)} batches, each size={len(batches[0])}")

miner = HardExampleMiner(n_samples=100, warmup_epochs=0)
miner.update_losses(list(range(10)), [float(i) for i in range(10)])
w = miner.get_sample_weights()
print(f"HardExampleMiner: weights shape={w.shape}, is_active={miner.is_active}")

# Test error analysis
from src.training.error_analysis import ErrorAnalyzer
analyzer = ErrorAnalyzer()
report = analyzer.analyze(y_true, y_probs, fold_idx=1)
fn_count = report["summary"]["FN"]
fp_count = report["summary"]["FP"]
print(f"ErrorAnalyzer: FN={fn_count}, FP={fp_count}")

# Test augmentor
from src.training.augmentation import PhysiologicalAugmentor
augmentor = PhysiologicalAugmentor(p=1.0)
augmentor.train()
X = torch.randn(4, 2, 4800)
X_aug = augmentor(X)
assert X_aug.shape == (4, 2, 4800)
print("PhysiologicalAugmentor: shape preserved [OK]")

print("\n[ALL PASSED] All Phase 4+ modules validated successfully.")
