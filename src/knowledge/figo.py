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
