"""Link-prediction anomaly detectors.

Both detectors in this sub-package implement the same `Detector`
interface (`base.py`) and score host-to-host edges with the convention
used everywhere else in this project: **higher score = more anomalous**
(so `lmsafe.metrics.recall_at_budget` can always just take the top-B
scores as "alerts", for any detector).

count_baseline.py  Non-parametric frequency baseline. No torch
                    dependency -- always available, always fast, and a
                    meaningful RQ1/RQ2 reference point: if a fancier
                    model cannot beat counting how often an edge has
                    been seen before, the fancier model is not earning
                    its complexity.
feature_mlp.py      A small torch MLP link-predictor over *running,
                     incrementally-updated per-host features* (degree,
                     partner diversity, recency) rather than a per-node
                     ID embedding table. This is a deliberate choice: an
                     ID-embedding table cannot score a host it never saw
                     during training (the exact non-inductive failure
                     mode this project's literature review flags in
                     CyberGFM's node-ID tokenization). Feature-based
                     scoring degrades gracefully to "unknown host, cold
                     features, high uncertainty" instead of crashing or
                     silently mis-scoring.
rw_gnn.py           This project's "graph foundation model": the same
                     inductive per-host features as feature_mlp.py, plus
                     one round of GraphSAGE-style neighborhood
                     aggregation within each window's graph, trained via
                     self-supervised link prediction, with weights that
                     can be pretrained once on one network (see
                     scripts/pretrain_foundation_model.py) and reused --
                     zero-shot or fine-tuned -- on a different network's
                     windows. See the module docstring for the precise,
                     deliberately modest scope claimed by "foundation
                     model" here.
"""
from .base import Detector
from .count_baseline import CountBaselineDetector
from .feature_mlp import FeatureMLPDetector
from .rw_gnn import GraphFoundationDetector

__all__ = ["Detector", "CountBaselineDetector", "FeatureMLPDetector", "GraphFoundationDetector"]
