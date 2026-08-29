import numpy as np

def build_topk_candidate_set(pred_scores, k=5):
    """
    Construct fixed Top-k candidate set for each sample.
    pred_scores: (N, 256) array of logits or power profiles.
    Returns: list of sets/arrays of length N.
    """
    topk_indices = np.argsort(pred_scores, axis=1)[:, -k:]
    return topk_indices

def build_threshold_candidate_set(pred_profile_db, threshold_q_db):
    """
    Construct power-aware threshold candidate set:
    C_q(x) = {b : P_hat(b) >= max_b'(P_hat(b')) - q}
    pred_profile_db: (N, 256)
    threshold_q_db: scalar float
    Returns: list of 1D numpy arrays (variable length candidate sets).
    """
    max_preds = np.max(pred_profile_db, axis=1, keepdims=True)
    mask = (pred_profile_db >= (max_preds - threshold_q_db))
    candidate_sets = [np.where(mask[i])[0] for i in range(len(pred_profile_db))]
    return candidate_sets

def evaluate_candidate_set_risk(candidate_sets, true_profile_db, delta_db=3.0):
    """
    Evaluate empirical miss rate (risk) and power loss for a list of candidate sets:
    Loss = 1 if max_{b in C} P(b) < max_b P(b) - delta_db, else 0.
    """
    N = len(candidate_sets)
    losses = np.zeros(N, dtype=np.float32)
    power_losses_db = np.zeros(N, dtype=np.float32)
    set_sizes = np.array([len(c) for c in candidate_sets], dtype=np.float32)

    best_powers = np.max(true_profile_db, axis=1)

    for i in range(N):
        c = candidate_sets[i]
        if len(c) == 0:
            losses[i] = 1.0
            power_losses_db[i] = best_powers[i] - np.min(true_profile_db[i])
        else:
            cand_max_pwr = np.max(true_profile_db[i, c])
            gap = best_powers[i] - cand_max_pwr
            power_losses_db[i] = gap
            losses[i] = 1.0 if (gap > delta_db) else 0.0

    return {
        "miss_rate": float(np.mean(losses)),
        "avg_size": float(np.mean(set_sizes)),
        "median_size": float(np.median(set_sizes)),
        "avg_power_loss_db": float(np.mean(power_losses_db)),
        "losses": losses,
        "set_sizes": set_sizes,
        "power_losses_db": power_losses_db
    }

class StaticConformalRiskControl:
    """
    Static Conformal Risk Control (CRC) calibrated on the calibration split.
    Guarantees E[Loss] <= alpha on exchangeable test samples.
    """
    def __init__(self, target_alpha=0.10, delta_db=3.0, q_grid=None):
        self.target_alpha = target_alpha
        self.delta_db = delta_db
        if q_grid is None:
            self.q_grid = np.linspace(0.0, 30.0, 301)  # 0.1 dB step
        else:
            self.q_grid = q_grid
        self.calib_q = None

    def fit(self, calib_pred_profile_db, calib_true_profile_db):
        n_calib = len(calib_pred_profile_db)
        best_q = self.q_grid[-1]

        for q in self.q_grid:
            cand_sets = build_threshold_candidate_set(calib_pred_profile_db, q)
            res = evaluate_candidate_set_risk(cand_sets, calib_true_profile_db, self.delta_db)
            emp_risk = res["miss_rate"]
            # Conformal adjusted bound
            adj_risk = (n_calib / (n_calib + 1.0)) * emp_risk + (1.0 / (n_calib + 1.0))
            if adj_risk <= self.target_alpha:
                best_q = q
                break

        self.calib_q = float(best_q)
        print(f"Static CRC calibrated q: {self.calib_q:.2f} dB (target alpha = {self.target_alpha}, delta = {self.delta_db} dB)")
        return self.calib_q

    def predict(self, test_pred_profile_db, test_true_profile_db):
        if self.calib_q is None:
            raise ValueError("StaticConformalRiskControl must be fitted on calibration set before prediction.")
        cand_sets = build_threshold_candidate_set(test_pred_profile_db, self.calib_q)
        metrics = evaluate_candidate_set_risk(cand_sets, test_true_profile_db, self.delta_db)
        metrics["calib_q"] = self.calib_q
        metrics["target_alpha"] = self.target_alpha
        metrics["delta_db"] = self.delta_db
        return cand_sets, metrics


class ExactLabelCRC:
    """
    Split conformal risk control on exact best-beam labels (Phase 6 Core).
    Nonconformity = 1 - softmax probability of the true class.
    """

    def __init__(self, target_alpha=0.10):
        self.target_alpha = target_alpha
        self.calib_q = None

    def fit(self, calib_logits, calib_labels):
        probs = _softmax(calib_logits)
        n = len(calib_labels)
        scores = 1.0 - probs[np.arange(n), calib_labels]
        level = np.ceil((n + 1) * (1.0 - self.target_alpha)) / n
        level = min(max(level, 0.0), 1.0)
        self.calib_q = float(np.quantile(scores, min(max(level, 0.0), 1.0)))
        print(f"Exact-label CRC quantile q={self.calib_q:.4f} (alpha={self.target_alpha})")
        return self.calib_q

    def predict(self, test_logits):
        if self.calib_q is None:
            raise ValueError("ExactLabelCRC must be fitted first.")
        probs = _softmax(test_logits)
        sets = [np.where(1.0 - probs[i] <= self.calib_q)[0] for i in range(len(probs))]
        return sets


def _softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def evaluate_fixed_topk(pred_scores, true_labels, true_profile_db, ks=(1, 3, 5, 10, 15), delta_db=3.0):
    rows = []
    for k in ks:
        sets = build_topk_candidate_set(pred_scores, k=k)
        list_sets = [sets[i] for i in range(len(sets))]
        risk = evaluate_candidate_set_risk(list_sets, true_profile_db, delta_db=delta_db)
        hits = np.mean([true_labels[i] in list_sets[i] for i in range(len(true_labels))])
        rows.append({
            "method": f"top{k}",
            "k_or_q": k,
            "exact_inclusion": float(hits),
            "miss_rate_delta_{}db".format(delta_db): risk["miss_rate"],
            "avg_size": risk["avg_size"],
            "avg_power_loss_db": risk["avg_power_loss_db"],
            "search_reduction_vs_256": float(1.0 - risk["avg_size"] / 256.0),
        })
    return rows

