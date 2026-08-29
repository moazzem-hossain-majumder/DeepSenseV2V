import numpy as np

def run_online_aci_controller(
    pred_profile_db,
    true_profile_db,
    q_init,
    target_alpha=0.10,
    eta=0.01,
    delta_db=3.0,
    q_min=0.0,
    q_max=30.0
):
    """
    Adaptive Conformal Inference (ACI) / Integral Controller over a streaming sequence.
    Updates threshold dynamically: q_{t+1} = clip(q_t + eta * (err_t - alpha), q_min, q_max).
    """
    T = len(pred_profile_db)
    q_history = np.zeros(T, dtype=np.float32)
    err_history = np.zeros(T, dtype=np.float32)
    size_history = np.zeros(T, dtype=np.float32)
    pwr_loss_history = np.zeros(T, dtype=np.float32)

    current_q = float(q_init)
    best_true_powers = np.max(true_profile_db, axis=1)

    for t in range(T):
        q_history[t] = current_q
        pred_pwr = pred_profile_db[t]
        true_pwr = true_profile_db[t]
        max_pred = np.max(pred_pwr)

        # 1. Construct candidate set at time t
        cand_set = np.where(pred_pwr >= (max_pred - current_q))[0]
        size_history[t] = len(cand_set)

        # 2. Observe ground truth and compute loss
        if len(cand_set) == 0:
            err = 1.0
            pwr_loss = best_true_powers[t] - np.min(true_pwr)
        else:
            cand_max_pwr = np.max(true_pwr[cand_set])
            pwr_loss = best_true_powers[t] - cand_max_pwr
            err = 1.0 if (pwr_loss > delta_db) else 0.0

        err_history[t] = err
        pwr_loss_history[t] = pwr_loss

        # 3. Update threshold for t+1
        current_q = np.clip(current_q + eta * (err - target_alpha), q_min, q_max)

    # Compute rolling miss rate over 100-step window
    window = min(100, T)
    rolling_miss = np.convolve(err_history, np.ones(window)/window, mode="valid")

    return {
        "miss_rate": float(np.mean(err_history)),
        "avg_size": float(np.mean(size_history)),
        "median_size": float(np.median(size_history)),
        "avg_power_loss_db": float(np.mean(pwr_loss_history)),
        "q_history": q_history,
        "err_history": err_history,
        "size_history": size_history,
        "rolling_miss": rolling_miss,
        "eta": eta,
        "target_alpha": target_alpha,
        "delta_db": delta_db
    }

def select_best_eta_on_val(val_pred_profile, val_true_profile, q_init, target_alpha=0.10, delta_db=3.0):
    """
    Grid-search optimal eta parameter on Validation split.
    Selects eta minimizing deviation from target_alpha while maintaining minimum candidate set size.
    """
    candidate_etas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    best_eta = candidate_etas[0]
    best_score = float("inf")

    print("\n[Tuning ACI Controller on Validation Split]")
    for eta in candidate_etas:
        res = run_online_aci_controller(
            val_pred_profile, val_true_profile,
            q_init=q_init, target_alpha=target_alpha, eta=eta, delta_db=delta_db
        )
        # Score penalizes miss rate deviation from alpha + normalized set size
        alpha_dev = abs(res["miss_rate"] - target_alpha)
        score = alpha_dev * 10.0 + (res["avg_size"] / 256.0)
        print(f"  eta={eta:6.3f} -> Miss Rate: {res['miss_rate']:.4f} (target {target_alpha}), Avg Size: {res['avg_size']:.2f}")
        if score < best_score:
            best_score = score
            best_eta = eta

    print(f"Selected optimal eta = {best_eta}")
    return best_eta


def run_online_pid_controller(
    pred_profile_db,
    true_profile_db,
    q_init,
    target_alpha=0.10,
    eta_p=0.1,
    eta_i=0.01,
    eta_d=0.01,
    delta_db=3.0,
    q_min=0.0,
    q_max=30.0,
):
    """
    Conformal PID Control (Angelopoulos, Candès, Tibshirani 2023) applied to
    near-optimal-beam miss risk. Stretch Phase 7 comparison vs ACI/integral.
    """
    T = len(pred_profile_db)
    q_history = np.zeros(T, dtype=np.float32)
    err_history = np.zeros(T, dtype=np.float32)
    size_history = np.zeros(T, dtype=np.float32)
    pwr_loss_history = np.zeros(T, dtype=np.float32)

    current_q = float(q_init)
    integral = 0.0
    prev_err = 0.0
    best_true_powers = np.max(true_profile_db, axis=1)

    for t in range(T):
        q_history[t] = current_q
        pred_pwr = pred_profile_db[t]
        true_pwr = true_profile_db[t]
        max_pred = np.max(pred_pwr)
        cand_set = np.where(pred_pwr >= (max_pred - current_q))[0]
        size_history[t] = len(cand_set)
        if len(cand_set) == 0:
            err = 1.0
            pwr_loss = best_true_powers[t] - np.min(true_pwr)
        else:
            cand_max_pwr = np.max(true_pwr[cand_set])
            pwr_loss = best_true_powers[t] - cand_max_pwr
            err = 1.0 if (pwr_loss > delta_db) else 0.0
        err_history[t] = err
        pwr_loss_history[t] = pwr_loss
        residual = err - target_alpha
        integral += residual
        deriv = residual - (prev_err - target_alpha)
        current_q = np.clip(current_q + eta_p * residual + eta_i * integral + eta_d * deriv, q_min, q_max)
        prev_err = err

    window = min(100, T)
    rolling_miss = np.convolve(err_history, np.ones(window) / window, mode="valid")
    return {
        "miss_rate": float(np.mean(err_history)),
        "avg_size": float(np.mean(size_history)),
        "median_size": float(np.median(size_history)),
        "avg_power_loss_db": float(np.mean(pwr_loss_history)),
        "q_history": q_history,
        "err_history": err_history,
        "size_history": size_history,
        "rolling_miss": rolling_miss,
        "eta_p": eta_p,
        "eta_i": eta_i,
        "eta_d": eta_d,
        "target_alpha": target_alpha,
        "delta_db": delta_db,
        "controller": "Conformal-PID",
    }
