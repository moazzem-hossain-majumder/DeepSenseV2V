"""
Kinematically Modulated Adaptive Conformal Controller (KM-ACC) for Vehicular mmWave Links.

Extends Adaptive Conformal Inference (Gibbs & Candès 2021) and Conformal PID (Angelopoulos et al. 2023)
by incorporating real-time vehicle turning kinematics as a feedforward modulation factor:
- Under steady highway driving (|d_theta/dt| ~ 0), candidate sets stay compact (~2-4 beams).
- Under dynamic urban turns (|d_theta/dt| > 0.1 rad/s), the threshold expands predictively,
  preventing consecutive outage bursts before feedback arrival.
"""

import numpy as np


class KinematicAdaptiveConformalController:
    """
    Trajectory-Adaptive Conformal Controller with Kinematic Feedforward.
    """
    def __init__(
        self,
        target_alpha=0.10,
        delta_db=3.0,
        eta_p=0.05,
        eta_i=0.01,
        eta_d=0.01,
        kappa_turning=1.5,
        q_min=0.0,
        q_max=20.0,
    ):
        self.target_alpha = target_alpha
        self.delta_db = delta_db
        self.eta_p = eta_p
        self.eta_i = eta_i
        self.eta_d = eta_d
        self.kappa_turning = kappa_turning
        self.q_min = q_min
        self.q_max = q_max

    def run_stream(
        self,
        pred_profiles_db,
        true_profiles_db,
        q_init,
        turning_rates_rad_s=None,
    ):
        """
        Executes online controller over a streaming vehicular sequence.

        pred_profiles_db: (T, 256) predicted relative power profiles in dB
        true_profiles_db: (T, 256) measured ground-truth power profiles in dB
        q_init: initial calibrated threshold in dB
        turning_rates_rad_s: (T,) optional vehicle yaw rate in rad/s

        Returns comprehensive diagnostic dictionary.
        """
        T = len(pred_profiles_db)
        q_history = np.zeros(T, dtype=np.float32)
        err_history = np.zeros(T, dtype=np.float32)
        size_history = np.zeros(T, dtype=np.float32)
        pwr_loss_history = np.zeros(T, dtype=np.float32)

        current_q = float(q_init)
        integral_err = 0.0
        prev_err = 0.0

        best_true_powers = np.max(true_profiles_db, axis=1)

        for t in range(T):
            # Apply kinematic feedforward if turning rate available
            if turning_rates_rad_s is not None and t < len(turning_rates_rad_s):
                yaw_rate = abs(float(turning_rates_rad_s[t]))
                kinematic_boost = self.kappa_turning * yaw_rate
            else:
                kinematic_boost = 0.0

            effective_q = np.clip(current_q + kinematic_boost, self.q_min, self.q_max)
            q_history[t] = effective_q

            # 1. Construct candidate set at time t
            pred_pwr = pred_profiles_db[t]
            true_pwr = true_profiles_db[t]
            max_pred = np.max(pred_pwr)

            cand_set = np.where(pred_pwr >= (max_pred - effective_q))[0]
            size_history[t] = len(cand_set)

            # 2. Observe ground truth power & calculate miss event
            if len(cand_set) == 0:
                err = 1.0
                pwr_loss = best_true_powers[t] - np.min(true_pwr)
            else:
                cand_max_pwr = np.max(true_pwr[cand_set])
                pwr_loss = best_true_powers[t] - cand_max_pwr
                err = 1.0 if (pwr_loss > self.delta_db) else 0.0

            err_history[t] = err
            pwr_loss_history[t] = pwr_loss

            # 3. PID error signals
            residual = err - self.target_alpha
            integral_err += residual
            # Anti-windup clamping on integral term
            integral_err = np.clip(integral_err, -50.0, 50.0)
            derivative_err = err - prev_err
            prev_err = err

            # 4. Controller update for t+1
            pid_delta = (
                self.eta_p * residual +
                self.eta_i * integral_err +
                self.eta_d * derivative_err
            )
            current_q = np.clip(current_q + pid_delta, self.q_min, self.q_max)

        # Rolling miss rate over 50-step window
        window = min(50, T)
        rolling_miss = np.convolve(err_history, np.ones(window) / window, mode="valid")

        return {
            "miss_rate": float(np.mean(err_history)),
            "target_alpha": self.target_alpha,
            "avg_candidate_size": float(np.mean(size_history)),
            "median_candidate_size": float(np.median(size_history)),
            "search_space_reduction_pct": float(100.0 * (1.0 - np.mean(size_history) / 256.0)),
            "avg_power_loss_db": float(np.mean(pwr_loss_history)),
            "q_history": q_history,
            "err_history": err_history,
            "size_history": size_history,
            "rolling_miss": rolling_miss,
            "pwr_loss_history": pwr_loss_history,
        }


def evaluate_trajectory_block_control(
    pred_profiles,
    true_profiles,
    seq_indices,
    q_init,
    controller=None,
):
    """
    Evaluates online controller per trajectory block, tracking worst-case trajectory risk.
    """
    if controller is None:
        controller = KinematicAdaptiveConformalController()

    unique_seqs = np.unique(seq_indices)
    block_results = []

    for seq in unique_seqs:
        mask = (seq_indices == seq)
        sub_pred = pred_profiles[mask]
        sub_true = true_profiles[mask]

        if len(sub_pred) < 10:
            continue

        res = controller.run_stream(sub_pred, sub_true, q_init=q_init)
        block_results.append({
            "seq_index": int(seq),
            "n_samples": len(sub_pred),
            "miss_rate": res["miss_rate"],
            "avg_size": res["avg_candidate_size"],
            "avg_loss_db": res["avg_power_loss_db"],
        })

    all_miss_rates = [b["miss_rate"] for b in block_results]
    all_sizes = [b["avg_size"] for b in block_results]
    all_losses = [b["avg_loss_db"] for b in block_results]

    return {
        "n_trajectories": len(block_results),
        "mean_trajectory_miss_rate": float(np.mean(all_miss_rates)),
        "worst_trajectory_miss_rate": float(np.max(all_miss_rates)),
        "best_trajectory_miss_rate": float(np.min(all_miss_rates)),
        "mean_candidate_size": float(np.mean(all_sizes)),
        "mean_power_loss_db": float(np.mean(all_losses)),
        "trajectory_details": block_results,
    }
