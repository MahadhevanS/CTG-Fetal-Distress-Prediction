from typing import Dict

def classify_figo(baseline: float, variability_ltv: float, 
                  accelerations: int, decelerations: Dict[str, int]) -> int:
    """
    Classifies a CTG window into FIGO categories (2015 guidelines simplified).
    0 = Normal, 1 = Suspicious, 2 = Pathological
    
    Args:
        baseline: Baseline FHR in bpm.
        variability_ltv: Long-term variability amplitude.
        accelerations: Number of accelerations.
        decelerations: Dictionary of deceleration counts ('early', 'late', 'variable', 'prolonged').
        
    Returns:
        int: FIGO class (0, 1, or 2).
    """
    # 1. Baseline Rules
    normal_baseline = 110 <= baseline <= 160
    
    # 2. Variability Rules
    normal_var = 5 <= variability_ltv <= 25
    reduced_var = variability_ltv < 5
    increased_var = variability_ltv > 25
    
    # 3. Deceleration Rules
    has_prolonged = decelerations.get('prolonged', 0) > 0
    has_late = decelerations.get('late', 0) > 0
    has_variable = decelerations.get('variable', 0) > 0
    
    # Simple FIGO logic mapping
    # Pathological: baseline < 100, reduced var > 50 min (simplified here), 
    #               repeated late/prolonged decels.
    if baseline < 100 or has_prolonged or (has_late and reduced_var):
        return 2 # Pathological
        
    # Suspicious: lacking one normal characteristic, but not pathological
    if not normal_baseline or not normal_var or has_late or has_variable:
        return 1 # Suspicious
        
    # Normal: baseline 110-160, var 5-25, no late/prolonged decels
    return 0 # Normal


import torch
import torch.nn as nn
import torch.nn.functional as F

def figo_rule_loss(pred_features: torch.Tensor, 
                   pred_figo_logits: torch.Tensor, 
                   target_figo: torch.Tensor = None, 
                   lambda_consistency: float = 0.5) -> torch.Tensor:
    """
    Computes the Knowledge-Infused FIGO Loss for multi-task training.
    
    Args:
        pred_features: Tensor of shape (Batch, 8) containing predicted clinical features:
                       [Baseline, STV, LTV, Accel_Count, Early_Decel, Late_Decel, Var_Decel, Prolonged_Decel]
        pred_figo_logits: Tensor of shape (Batch, 3) containing 3-class logits for FIGO (Normal=0, Suspicious=1, Pathological=2).
        target_figo: Optional Tensor of shape (Batch,) with ground-truth FIGO class targets (0, 1, 2).
        lambda_consistency: Weight factor for soft clinical consistency penalties.
        
    Returns:
        torch.Tensor: Scalar loss combining cross-entropy and FIGO clinical rule penalties.
    """
    total_loss = torch.tensor(0.0, device=pred_figo_logits.device)
    
    # 1. Primary Classification Loss (if target provided)
    if target_figo is not None:
        ce_loss = F.cross_entropy(pred_figo_logits, target_figo.long())
        total_loss = total_loss + ce_loss
        
    # Softmax probabilities for FIGO classes
    probs = F.softmax(pred_figo_logits, dim=-1)
    p_normal = probs[:, 0]
    p_pathological = probs[:, 2]
    
    # Extract predicted clinical feature columns (8-column mapping)
    baseline = pred_features[:, 0]
    stv = pred_features[:, 1]
    ltv = pred_features[:, 2]
    late_decels = pred_features[:, 5]
    prolonged_decels = pred_features[:, 7]
    
    # 2. Differentiable Rule Consistency Penalties
    
    # Rule A: Baseline Deviation Penalty
    # If baseline < 110 or > 160, penalty increases if network predicts Normal (p_normal -> 1)
    baseline_under = F.relu(110.0 - baseline)
    baseline_over = F.relu(baseline - 160.0)
    penalty_baseline = (baseline_under**2 + baseline_over**2) * p_normal
    
    # Rule B: Variability Deviation Penalty
    # If LTV < 5 or > 25, penalty increases if network predicts Normal (p_normal -> 1)
    ltv_under = F.relu(5.0 - ltv)
    ltv_over = F.relu(ltv - 25.0)
    penalty_ltv = (ltv_under**2 + ltv_over**2) * p_normal
    
    # Rule C: Pathological Deceleration Penalty
    # Late or prolonged decelerations should penalize Normal predictions
    penalty_decels = (F.relu(late_decels) + 2.0 * F.relu(prolonged_decels)) * p_normal
    
    # Rule D: Pathological Baseline Penalty
    # Baseline < 100 or prolonged decels present should penalize non-Pathological predictions (1 - p_pathological)
    patho_baseline = F.relu(100.0 - baseline)
    penalty_patho = (patho_baseline + F.relu(prolonged_decels)) * (1.0 - p_pathological)
    
    # Sum and normalize consistency loss across batch
    consistency_loss = (penalty_baseline + penalty_ltv + penalty_decels + penalty_patho).mean()
    
    total_loss = total_loss + lambda_consistency * consistency_loss
    return total_loss

