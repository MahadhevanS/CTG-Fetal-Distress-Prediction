"""
FIGO-state detection and early-deterioration prediction.

A separate track from the pH-outcome work in src/preprocessing/ and
src/models/. Nothing here imports or mutates that pipeline, so every existing
result stays reproducible (repo convention, see pipeline_clinical.py).

Modules
-------
descriptors  per-epoch FIGO characteristics from raw FHR + UC
rules        the FROZEN FIGO 2015 state classifier (label definition)
epochs       non-overlapping epoch segmentation + quality gating
"""
