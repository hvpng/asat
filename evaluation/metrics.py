# evaluation/metrics.py
# Tính các chỉ số đánh giá: AUA và S_adv.
#
# Không biết gì về model, dataset, hay attack — chỉ nhận số liệu đã tính.
# Dùng lại ở mọi tuần (baseline, FreeLB, ASAT).
#
# API:
#   from evaluation.metrics import compute_aua, compute_s_adv

import numpy as np
from scipy.stats import spearmanr


def compute_aua(attack_results):
    """
    Accuracy Under Attack (↑).

    AUA = tỷ lệ mẫu mà prediction KHÔNG bị flip sau attack.
    = (n_failed + n_skipped) / n_total

    Args:
        attack_results: list of dict với key "pred_flipped" (bool)
                        — output từ attack_eval.py

    Returns:
        aua     : float
        n_total : int
        n_flipped: int
    """
    n_total   = len(attack_results)
    n_flipped = sum(1 for r in attack_results if r["pred_flipped"])
    aua = (n_total - n_flipped) / max(n_total, 1)
    return aua, n_total, n_flipped


def compute_s_adv(attr_pairs):
    """
    Adversarial Sensitivity S_adv (↓).

    S_adv = 1 - mean( Spearman(IG_clean_i, IG_attacked_i) )
    Chỉ tính trên label-stable samples (pred_flipped=False).

    Args:
        attr_pairs: list of dict:
            {
                "attr_clean"   : np.ndarray [seq_len],  # IG clean
                "attr_attacked": np.ndarray [seq_len],  # IG attacked
                "pred_flipped" : bool,
            }

    Returns:
        s_adv          : float — NaN nếu không có label-stable sample nào
        n_stable       : int   — số mẫu label-stable được dùng
        spearman_values: list[float] — correlation từng mẫu (để bootstrap)
    """
    stable = [p for p in attr_pairs if not p["pred_flipped"]]
    n_stable = len(stable)

    if n_stable == 0:
        return float("nan"), 0, []

    spearman_values = []
    for p in stable:
        rho, _ = spearmanr(p["attr_clean"], p["attr_attacked"])
        spearman_values.append(float(rho))

    s_adv = 1.0 - float(np.mean(spearman_values))
    return s_adv, n_stable, spearman_values