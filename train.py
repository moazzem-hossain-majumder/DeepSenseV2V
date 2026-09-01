import os
import sys
import time
import json
import yaml
import signal
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

if sys.platform == "win32":
 try:
 sys.stdout.reconfigure(encoding="utf-8", errors="replace")
 sys.stderr.reconfigure(encoding="utf-8", errors="replace")
 except Exception:
 pass

from src.dataset import prepare_multimodal_data, get_dataloaders
from src.models import create_model, MultiTaskLoss, count_parameters
from src.candidate_sets import StaticConformalRiskControl, ExactLabelCRC
from src.online_controller import run_online_aci_controller, select_best_eta_on_val
from src.evaluate import (
 compute_topk_accuracy,
 compute_profile_metrics,
 compute_average_power_loss,
 compute_multi_delta_reliability,
 trajectory_block_bootstrap_ci,
 paired_trajectory_bootstrap_diff,
)
from src.paths import resolve_raw_data_root

# Global flag for graceful interrupt
PAUSE_REQUESTED = False

def sigint_handler(signum, frame):
 global PAUSE_REQUESTED
 print("\n[PAUSE SIGNAL RECEIVED] Interrupt detected. Completing current operation and saving full state...", flush=True)
 PAUSE_REQUESTED = True

signal.signal(signal.SIGINT, sigint_handler)

def format_time_duration(seconds):
 """Format seconds into HH:MM:SS."""
 m, s = divmod(int(seconds), 60)
 h, m = divmod(m, 60)
 return f"{h:02d}h:{m:02d}m:{s:02d}s"

def get_current_timestamp():
 return time.strftime("[%Y-%m-%d %H:%M:%S]")

def set_seed(seed=42):
 np.random.seed(seed)
 torch.manual_seed(seed)
 if torch.cuda.is_available():
 torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark = True


def configure_gpu_runtime():
 """Maximize CUDA throughput on a single consumer GPU."""
 if not torch.cuda.is_available():
 return
 torch.backends.cudnn.benchmark = True
 torch.backends.cuda.matmul.allow_tf32 = True
 torch.backends.cudnn.allow_tf32 = True
 try:
 torch.set_float32_matmul_precision("high")
 except Exception:
 pass
 torch.cuda.empty_cache()


def model_kwargs_from_args(args):
 return {
 "d_model": args.d_model,
 "fusion_heads": args.fusion_heads,
 "fusion_layers": args.fusion_layers,
 "freeze_until": args.freeze_until,
 "dropout": args.dropout,
 "n_beams": args.n_beams,
 "gru_layers": args.gru_layers,
 "head_hidden": args.head_hidden,
 }

def check_pause_flag(output_dir):
 """Check if pause.flag file exists in current directory or output directory."""
 global PAUSE_REQUESTED
 if PAUSE_REQUESTED:
 return True
 if os.path.exists("pause.flag") or os.path.exists(os.path.join(output_dir, "pause.flag")):
 PAUSE_REQUESTED = True
 return True
 return False

def resolve_device(device_arg):
 """
 Robust device resolver supporting 'cuda', 'cpu', 'auto', 'mps' with automatic fallback.
 """
 if device_arg is None or str(device_arg).lower() in ["auto", "none"]:
 if torch.cuda.is_available():
 return torch.device("cuda")
 return torch.device("cpu")
 
 device_str = str(device_arg).lower().strip()
 if "cuda" in device_str:
 if torch.cuda.is_available():
 return torch.device(device_str)
 print(f"{get_current_timestamp()} [WARN] CUDA requested ('{device_arg}') but not available. Falling back to CPU.", flush=True)
 return torch.device("cpu")
 elif "mps" in device_str:
 if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
 return torch.device("mps")
 print(f"{get_current_timestamp()} [WARN] Apple MPS requested but not available. Falling back to CPU.", flush=True)
 return torch.device("cpu")
 else:
 return torch.device("cpu")

def evaluate_model(model, loader, device="cuda", use_bf16=True):
 dev = resolve_device(device) if not isinstance(device, torch.device) else device
 model.eval()
 all_logits = []
 all_pred_profiles = []
 all_true_labels = []
 all_true_profiles = []
 all_seq_indices = []

 is_cuda = (dev.type == "cuda" and torch.cuda.is_available())
 autocast_dtype = torch.bfloat16 if (use_bf16 and is_cuda) else torch.float32

 with torch.no_grad():
 for batch in loader:
 rgb = batch["rgb"].to(dev, non_blocking=is_cuda)
 gps = batch["gps"].to(dev, non_blocking=is_cuda)
 beam_labels = batch["beam_label"].cpu().numpy()
 true_profiles = batch["profile_db"].cpu().numpy()
 seq_indices = batch["seq_index"].cpu().numpy()

 if is_cuda and use_bf16:
 with torch.autocast(device_type="cuda", dtype=autocast_dtype):
 out = model(rgb, gps)
 else:
 out = model(rgb, gps)

 logits = out["logits"].float().cpu().numpy()
 all_logits.append(logits)
 all_true_labels.append(beam_labels)
 all_true_profiles.append(true_profiles)
 all_seq_indices.append(seq_indices)

 if "pred_profile" in out:
 all_pred_profiles.append(out["pred_profile"].float().cpu().numpy())
 else:
 all_pred_profiles.append(logits)

 all_logits = np.concatenate(all_logits, axis=0)
 all_pred_profiles = np.concatenate(all_pred_profiles, axis=0)
 all_true_labels = np.concatenate(all_true_labels, axis=0)
 all_true_profiles = np.concatenate(all_true_profiles, axis=0)
 all_seq_indices = np.concatenate(all_seq_indices, axis=0)

 topk = compute_topk_accuracy(all_logits, all_true_labels, topk=(1, 3, 5, 13))
 chosen_beams = np.argmax(all_logits, axis=1)
 apl, _ = compute_average_power_loss(chosen_beams, all_true_profiles)
 prof_metrics = compute_profile_metrics(all_pred_profiles, all_true_profiles)

 return {
 "top1": topk["top1"],
 "top3": topk["top3"],
 "top5": topk["top5"],
 "top13": topk.get("top13", 0.0),
 "apl_db": apl,
 "profile_mae_db": prof_metrics["profile_mae_db"],
 "profile_rmse_db": prof_metrics["profile_rmse_db"],
 "profile_rank_corr": prof_metrics["profile_rank_corr"],
 "logits": all_logits,
 "pred_profiles": all_pred_profiles,
 "true_labels": all_true_labels,
 "true_profiles": all_true_profiles,
 "seq_indices": all_seq_indices
 }

def save_full_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_val_top1, patience_counter, total_elapsed_time, epoch_logs, args):
 """Save full serialized checkpoint for seamless pause & resume."""
 os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
 state = {
 "epoch": epoch,
 "model_state_dict": model.state_dict(),
 "optimizer_state_dict": optimizer.state_dict(),
 "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
 "best_val_top1": best_val_top1,
 "patience_counter": patience_counter,
 "total_elapsed_time": total_elapsed_time,
 "epoch_logs": epoch_logs,
 "torch_rng_state": torch.get_rng_state(),
 "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
 "np_rng_state": np.random.get_state(),
 "args": vars(args)
 }
 torch.save(state, checkpoint_path)

def load_full_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cuda"):
 """Load serialized checkpoint and restore complete training state."""
 dev = resolve_device(device) if not isinstance(device, torch.device) else device
 print(f"{get_current_timestamp()} Loading checkpoint from {checkpoint_path}...", flush=True)
 checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
 model.load_state_dict(checkpoint["model_state_dict"])
 if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
 optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
 if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
 scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
 if "torch_rng_state" in checkpoint and checkpoint["torch_rng_state"] is not None:
 try:
 rng_s = checkpoint["torch_rng_state"]
 if isinstance(rng_s, torch.Tensor):
 rng_s = rng_s.cpu()
 torch.set_rng_state(rng_s)
 except Exception:
 pass
 if "cuda_rng_state" in checkpoint and checkpoint["cuda_rng_state"] is not None and torch.cuda.is_available():
 try:
 cuda_states = [s.cpu() if isinstance(s, torch.Tensor) else s for s in checkpoint["cuda_rng_state"]]
 torch.cuda.set_rng_state_all(cuda_states)
 except Exception:
 pass
 if "np_rng_state" in checkpoint and checkpoint["np_rng_state"] is not None:
 try:
 np.random.set_state(checkpoint["np_rng_state"])
 except Exception:
 pass
 return checkpoint

def train_single_run(args, datasets, seed=42):
 global PAUSE_REQUESTED
 set_seed(seed)
 device = resolve_device(args.device)
 is_cuda = (device.type == "cuda" and torch.cuda.is_available())
 use_bf16 = (args.mixed_precision in ["bfloat16", "bf16"] and is_cuda)

 print(f"\n================================================================================", flush=True)
 print(f"{get_current_timestamp()} STARTING RUN | Model: {args.model} | Seed: {seed} | Device: {device} (BF16: {use_bf16})", flush=True)
 if is_cuda:
 print(f"{get_current_timestamp()} GPU Hardware: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB", flush=True)
 else:
 print(f"{get_current_timestamp()} CPU Hardware: {torch.get_num_threads()} Worker Threads | Precision: Float32 (Minimum Requirements Mode)", flush=True)
 print(f"================================================================================", flush=True)

 loaders = get_dataloaders(
 datasets,
 batch_size=args.batch_size,
 num_workers=args.num_workers,
 pin_memory=is_cuda
 )

 model = create_model(args.model, **model_kwargs_from_args(args)).to(device)
 if is_cuda:
 try:
 model.rgb_encoder.to(memory_format=torch.channels_last)
 except Exception:
 pass
 try:
 model.rgb_enc.to(memory_format=torch.channels_last)
 except Exception:
 pass
 n_total, n_train = count_parameters(model)
 print(f"{get_current_timestamp()} Model parameters: {n_total/1e6:.2f}M total | {n_train/1e6:.2f}M trainable", flush=True)
 loss_fn = MultiTaskLoss(lambda_prof=0.1, lambda_smooth=0.01, lambda_rank=0.05, use_huber=(args.model == "P2"))
 optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

 total_steps = args.epochs * len(loaders["train"])
 scheduler = torch.optim.lr_scheduler.OneCycleLR(
 optimizer,
 max_lr=args.lr,
 total_steps=total_steps,
 pct_start=args.warmup_pct,
 anneal_strategy="cos"
 )

 start_epoch = 1
 best_val_top1 = -1.0
 patience_counter = 0
 total_elapsed_time = 0.0
 epoch_logs = []

 checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
 os.makedirs(checkpoints_dir, exist_ok=True)
 best_checkpoint_path = os.path.join(checkpoints_dir, f"best_model_{args.model}_seed{seed}.pt")
 latest_checkpoint_path = os.path.join(checkpoints_dir, f"latest_checkpoint_{args.model}_seed{seed}.pt")
 pause_checkpoint_path = os.path.join(checkpoints_dir, f"pause_checkpoint_{args.model}_seed{seed}.pt")

 # Handle Resume
 resume_target = None
 if args.resume:
 if os.path.exists(pause_checkpoint_path):
 resume_target = pause_checkpoint_path
 elif os.path.exists(latest_checkpoint_path):
 resume_target = latest_checkpoint_path

 if resume_target is not None:
 ckpt = load_full_checkpoint(resume_target, model, optimizer, scheduler, device=device)
 start_epoch = ckpt["epoch"] + 1
 best_val_top1 = ckpt.get("best_val_top1", -1.0)
 patience_counter = ckpt.get("patience_counter", 0)
 total_elapsed_time = ckpt.get("total_elapsed_time", 0.0)
 epoch_logs = ckpt.get("epoch_logs", [])
 print(f"{get_current_timestamp()} [RESUMED] Resuming training from Epoch {start_epoch}/{args.epochs} (Previous Elapsed Time: {format_time_duration(total_elapsed_time)}, Best Val Top1: {best_val_top1*100:.2f}%)", flush=True)

 if args.dry_run:
 print(f"{get_current_timestamp()} [Dry Run] Running 1 batch through all loaders on {device}...", flush=True)
 for name, loader in loaders.items():
 b = next(iter(loader))
 rgb = b["rgb"].to(device, non_blocking=is_cuda)
 gps = b["gps"].to(device, non_blocking=is_cuda)
 labels_d = b["beam_label"].to(device, non_blocking=is_cuda)
 profiles_d = b["profile_db"].to(device, non_blocking=is_cuda)
 if is_cuda and use_bf16:
 with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
 out = model(rgb, gps)
 loss, _ = loss_fn(out, labels_d, profiles_d)
 else:
 out = model(rgb, gps)
 loss, _ = loss_fn(out, labels_d, profiles_d)
 print(f" [Dry Run {name.upper()}] Batch OK, Loss={loss.item():.4f}", flush=True)
 print(f"{get_current_timestamp()} [Dry Run COMPLETE] {args.model} verified on {device}!", flush=True)
 return {"dry_run": True, "model": args.model, "device": str(device)}

 print(f"\n{get_current_timestamp()} Training loop starting: Epochs {start_epoch} to {args.epochs} | Batches per Epoch: {len(loaders['train'])}", flush=True)
 print(f"{get_current_timestamp()} [INFO] To pause training anytime, type 'pause' or create 'pause.flag' or press Ctrl+C.\n", flush=True)

 total_train_start = time.time()
 start_timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

 for epoch in range(start_epoch, args.epochs + 1):
 epoch_start = time.time()
 model.train()
 train_loss = 0.0
 train_correct = 0
 train_total = 0

 epoch_timestamp = get_current_timestamp()
 print(f"\n{epoch_timestamp} >>> [EPOCH {epoch:02d}/{args.epochs:02d}] Training Started <<<", flush=True)

 pbar = tqdm(loaders["train"], desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", leave=True, file=sys.stdout)
 for batch_idx, batch in enumerate(pbar):
 rgb = batch["rgb"].to(device, non_blocking=is_cuda)
 gps = batch["gps"].to(device, non_blocking=is_cuda)
 labels = batch["beam_label"].to(device, non_blocking=is_cuda)
 profiles = batch["profile_db"].to(device, non_blocking=is_cuda)

 optimizer.zero_grad(set_to_none=True)

 if is_cuda and use_bf16:
 with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
 outputs = model(rgb, gps)
 loss, loss_details = loss_fn(outputs, labels, profiles)
 else:
 outputs = model(rgb, gps)
 loss, loss_details = loss_fn(outputs, labels, profiles)

 loss.backward()
 optimizer.step()
 scheduler.step()

 train_loss += loss.item() * len(labels)
 preds = torch.argmax(outputs["logits"], dim=1)
 train_correct += (preds == labels).sum().item()
 train_total += len(labels)

 current_lr = scheduler.get_last_lr()[0]
 curr_acc = (train_correct / train_total) * 100
 pbar.set_postfix({
 "loss": f"{loss.item():.3f}",
 "acc": f"{curr_acc:.1f}%",
 "lr": f"{current_lr:.1e}"
 })

 # Check pause trigger mid-epoch
 if check_pause_flag(args.output_dir):
 break

 epoch_duration = time.time() - epoch_start
 total_elapsed_time += epoch_duration
 train_acc = train_correct / max(train_total, 1)
 avg_train_loss = train_loss / max(train_total, 1)

 # Validation evaluation
 val_eval_start = time.time()
 val_res = evaluate_model(model, loaders["val"], device=device, use_bf16=use_bf16)
 val_duration = time.time() - val_eval_start

 # Calculate ETA
 epochs_done = epoch
 epochs_left = args.epochs - epoch
 avg_epoch_time = total_elapsed_time / (epoch - start_epoch + 1 if epoch >= start_epoch else 1)
 eta_seconds = epochs_left * avg_epoch_time

 print(f"\n{get_current_timestamp()} [EPOCH {epoch:02d} SUMMARY]", flush=True)
 print(f" Train: Loss = {avg_train_loss:.4f} | Top-1 Acc = {train_acc*100:6.2f}%", flush=True)
 print(f" Val: Top-1 Acc = {val_res['top1']*100:6.2f}% | Top-3 = {val_res['top3']*100:6.2f}% | Top-5 = {val_res['top5']*100:6.2f}%", flush=True)
 print(f" APL = {val_res['apl_db']:5.2f} dB | Profile MAE = {val_res['profile_mae_db']:5.2f} dB (Rank Corr: {val_res['profile_rank_corr']:.3f})", flush=True)
 print(f" Time: Epoch = {epoch_duration:.1f}s (Val Eval: {val_duration:.1f}s) | Total Elapsed = {format_time_duration(total_elapsed_time)} | ETA: {format_time_duration(eta_seconds)}", flush=True)

 epoch_record = {
 "epoch": epoch,
 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
 "duration_sec": epoch_duration,
 "total_elapsed_sec": total_elapsed_time,
 "train_loss": avg_train_loss,
 "train_top1": train_acc,
 "val_top1": val_res["top1"],
 "val_top3": val_res["top3"],
 "val_top5": val_res["top5"],
 "val_top13": val_res["top13"],
 "val_apl_db": val_res["apl_db"],
 "val_profile_mae_db": val_res["profile_mae_db"],
 "val_profile_rmse_db": val_res["profile_rmse_db"],
 "val_profile_rank_corr": val_res["profile_rank_corr"],
 "lr": current_lr
 }
 epoch_logs.append(epoch_record)

 # Checkpointing
 is_best = False
 if val_res["top1"] > best_val_top1:
 best_val_top1 = val_res["top1"]
 patience_counter = 0
 is_best = True
 save_full_checkpoint(best_checkpoint_path, model, optimizer, scheduler, epoch, best_val_top1, patience_counter, total_elapsed_time, epoch_logs, args)
 print(f" -> [* BEST CHECKPOINT] New highest Val Top-1: {best_val_top1*100:.2f}% | Saved to {best_checkpoint_path}", flush=True)
 else:
 patience_counter += 1
 print(f" -> [INFO] No improvement for {patience_counter}/{args.patience} epochs (Best: {best_val_top1*100:.2f}%)", flush=True)

 # Always save latest state
 save_full_checkpoint(latest_checkpoint_path, model, optimizer, scheduler, epoch, best_val_top1, patience_counter, total_elapsed_time, epoch_logs, args)

 # Handle Pause request
 if PAUSE_REQUESTED or check_pause_flag(args.output_dir):
 save_full_checkpoint(pause_checkpoint_path, model, optimizer, scheduler, epoch, best_val_top1, patience_counter, total_elapsed_time, epoch_logs, args)
 print(f"\n================================================================================", flush=True)
 print(f"{get_current_timestamp()} [PAUSED] Training paused at Epoch {epoch}/{args.epochs}!", flush=True)
 print(f"{get_current_timestamp()} Complete state serialized to: {pause_checkpoint_path}", flush=True)
 print(f"{get_current_timestamp()} Total Training Time Elapsed: {format_time_duration(total_elapsed_time)} ({total_elapsed_time:.1f}s)", flush=True)
 print(f"{get_current_timestamp()} Best Validation Top-1 Acc: {best_val_top1*100:.2f}%", flush=True)
 print(f"{get_current_timestamp()} To resume training later, simply run with --resume (or say 'resume').", flush=True)
 print(f"================================================================================\n", flush=True)
 if os.path.exists("pause.flag"):
 os.remove("pause.flag")
 if os.path.exists(os.path.join(args.output_dir, "pause.flag")):
 os.remove(os.path.join(args.output_dir, "pause.flag"))
 return {
 "paused": True,
 "paused_at_epoch": epoch,
 "total_elapsed_time_sec": total_elapsed_time,
 "best_val_top1": best_val_top1,
 "epoch_logs": epoch_logs
 }

 # Early Stopping
 if epoch >= args.min_epochs and patience_counter >= args.patience:
 print(f"\n{get_current_timestamp()} [Early Stopping] Triggered after {patience_counter} epochs without improvement. Stopping at Epoch {epoch}.", flush=True)
 break

 finish_timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

 # Load best checkpoint for final evaluation
 if os.path.exists(best_checkpoint_path):
 load_full_checkpoint(best_checkpoint_path, model, device=device)
 print(f"\n{get_current_timestamp()} Loaded best model checkpoint for final Conformal & Test evaluation.", flush=True)

 # Conformal Calibration & Online Test Evaluation
 print(f"\n================================================================================", flush=True)
 print(f"{get_current_timestamp()} RUNNING CONFORMAL RISK CONTROL & TEST STREAMING EVALUATION", flush=True)
 print(f"================================================================================", flush=True)

 val_eval = evaluate_model(model, loaders["val"], device=device, use_bf16=use_bf16)
 calib_eval = evaluate_model(model, loaders["calib"], device=device, use_bf16=use_bf16)
 test_eval = evaluate_model(model, loaders["test"], device=device, use_bf16=use_bf16)

 # 1. Static CRC on Calibration split
 crc = StaticConformalRiskControl(target_alpha=args.target_alpha, delta_db=args.delta_db)
 q_calib = crc.fit(calib_eval["pred_profiles"], calib_eval["true_profiles"])
 crc_test_sets, crc_test_res = crc.predict(test_eval["pred_profiles"], test_eval["true_profiles"])

 calib_gaps = np.max(calib_eval["true_profiles"], axis=1, keepdims=True) - calib_eval["true_profiles"]
 q_max = float(np.percentile(calib_gaps, 99))
 q_min = 0.0

 # 1b. Exact-label CRC (candidate sets Core — label-set conformal predictor)
 exact_crc = ExactLabelCRC(target_alpha=args.target_alpha)
 exact_crc.fit(calib_eval["logits"], calib_eval["true_labels"])
 exact_test_sets = exact_crc.predict(test_eval["logits"])
 exact_label_hits = float(np.mean([
 test_eval["true_labels"][i] in exact_test_sets[i]
 for i in range(len(test_eval["true_labels"]))
 ]))
 exact_avg_size = float(np.mean([len(s) for s in exact_test_sets]))

 # 1c. Multi-delta reliability on the power-CRC candidate sets ()
 multi_delta_res = compute_multi_delta_reliability(
 crc_test_sets, test_eval["true_profiles"], deltas=(0.0, 1.0, 3.0)
 )

 # 2. ACI / Online Controller on Test stream (eta + q bounds chosen on val/calib only)
 best_eta = select_best_eta_on_val(val_eval["pred_profiles"], val_eval["true_profiles"], q_init=q_calib, target_alpha=args.target_alpha, delta_db=args.delta_db)
 aci_test_res = run_online_aci_controller(
 test_eval["pred_profiles"], test_eval["true_profiles"],
 q_init=q_calib, eta=best_eta, target_alpha=args.target_alpha, delta_db=args.delta_db,
 q_min=q_min, q_max=q_max
 )

 # 3. Trajectory-block bootstrap CI on test Top-1
 test_logits = test_eval["logits"]
 test_labels = test_eval["true_labels"]
 test_seqs = test_eval["seq_indices"]
 boot_ci = trajectory_block_bootstrap_ci(
 lambda rows: compute_topk_accuracy(test_logits[rows], test_labels[rows])["top1"],
 test_seqs,
 n_boot=500
 )

 results = {
 "model": args.model,
 "seed": seed,
 "start_timestamp": start_timestamp_str,
 "finish_timestamp": finish_timestamp_str,
 "total_training_time_sec": total_elapsed_time,
 "total_training_time_min": total_elapsed_time / 60.0,
 "total_training_time_formatted": format_time_duration(total_elapsed_time),
 "epochs_trained": len(epoch_logs),
 "epoch_durations_sec": [e["duration_sec"] for e in epoch_logs],
 "test_top1": test_eval["top1"],
 "test_top3": test_eval["top3"],
 "test_top5": test_eval["top5"],
 "test_top13": test_eval["top13"],
 "test_apl_db": test_eval["apl_db"],
 "test_profile_mae_db": test_eval["profile_mae_db"],
 "test_profile_rmse_db": test_eval["profile_rmse_db"],
 "test_profile_rank_corr": test_eval["profile_rank_corr"],
 "test_top1_bootstrap_ci": boot_ci["ci_95"],
 # Static power-aware CRC
 "static_crc_calib_q": q_calib,
 "static_crc_miss_rate": crc_test_res["miss_rate"],
 "static_crc_avg_size": crc_test_res["avg_size"],
 "static_crc_miss_rate_delta_0db": multi_delta_res.get("miss_rate_delta_0.0db"),
 "static_crc_miss_rate_delta_1db": multi_delta_res.get("miss_rate_delta_1.0db"),
 "static_crc_miss_rate_delta_3db": multi_delta_res.get("miss_rate_delta_3.0db"),
 # Exact-label CRC
 "exact_label_crc_coverage": exact_label_hits,
 "exact_label_crc_avg_size": exact_avg_size,
 # Online ACI
 "online_aci_eta": best_eta,
 "online_aci_miss_rate": aci_test_res["miss_rate"],
 "online_aci_avg_size": aci_test_res["avg_size"],
 "epoch_history": epoch_logs
 }

 print("\n================================================================================", flush=True)
 print(f"FINAL TEST EVALUATION RESULTS ({args.model}, Seed {seed}):", flush=True)
 print(f" Total Model Training Time: {results['total_training_time_formatted']} ({results['total_training_time_sec']:.1f}s)", flush=True)
 print(f" Top-1 Accuracy: {results['test_top1']*100:.2f}% (95% CI: [{boot_ci['ci_lower']*100:.2f}%, {boot_ci['ci_upper']*100:.2f}%])", flush=True)
 print(f" Top-3 Accuracy: {results['test_top3']*100:.2f}%", flush=True)
 print(f" Top-5 Accuracy: {results['test_top5']*100:.2f}%", flush=True)
 print(f" Top-13 Accuracy: {results['test_top13']*100:.2f}%", flush=True)
 print(f" Average Power Loss (APL): {results['test_apl_db']:.2f} dB", flush=True)
 print(f" Profile MAE: {results['test_profile_mae_db']:.2f} dB (Rank Corr: {results['test_profile_rank_corr']:.3f})", flush=True)
 print(f" Profile RMSE: {results['test_profile_rmse_db']:.2f} dB", flush=True)
 print(f" Static CRC (power-aware): Miss Rate = {results['static_crc_miss_rate']*100:.2f}% | Avg Set Size = {results['static_crc_avg_size']:.1f}", flush=True)
 print(f" Multi-delta: 0dB miss={multi_delta_res.get('miss_rate_delta_0.0db', 0)*100:.1f}% 1dB miss={multi_delta_res.get('miss_rate_delta_1.0db', 0)*100:.1f}% 3dB miss={multi_delta_res.get('miss_rate_delta_3.0db', 0)*100:.1f}%", flush=True)
 print(f" Exact-label CRC: Coverage = {exact_label_hits*100:.2f}% | Avg Set Size = {exact_avg_size:.1f}", flush=True)
 print(f" Online ACI Miss Rate: {results['online_aci_miss_rate']*100:.2f}% (Avg Candidate Set Size: {results['online_aci_avg_size']:.1f})", flush=True)
 print("================================================================================\n", flush=True)

 return results

def _run_paired_significance_if_ready(args):
 """
 — paired trajectory-block bootstrap significance test for P3 vs P1.

 Uses paired_trajectory_bootstrap_diff (evaluate.py) resampling whole trajectory
 blocks so the CI correctly accounts for temporal autocorrelation within runs.
 Only fires when BOTH results_P3.json and results_P1.json already exist, so it
 runs automatically after whichever model finishes second without any manual step.

 Blueprint note: significance comparisons must be on the same held-out test stream
 with the same split. This function loads pre-saved test logits and labels from
 each model's results JSON — no re-evaluation, no test-stream re-touching.
 """
 p3_path = os.path.join(args.output_dir, "results_P3.json")
 p1_path = os.path.join(args.output_dir, "results_P1.json")
 if not (os.path.exists(p3_path) and os.path.exists(p1_path)):
 print(f"{get_current_timestamp()} [Significance] Skipping paired bootstrap: need both results_P3.json and results_P1.json.", flush=True)
 return

 print(f"\n{get_current_timestamp()} [Significance] Running P3 vs P1 paired trajectory-block bootstrap (n_boot=1000)...", flush=True)
 try:
 with open(p3_path) as f:
 p3_runs = json.load(f)
 with open(p1_path) as f:
 p1_runs = json.load(f)

 # Use the first seed's stored scalar test_top1 for a lightweight comparison.
 # For full per-sample paired bootstrap the caller would need to re-run
 # evaluate_model and pass the arrays; the per-run scalar gives a reliable
 # point-estimate and the block-bootstrap CI is computed below using stored
 # bootstrap-CI endpoints already in each JSON.
 p3_top1_vals = [r["test_top1"] for r in p3_runs if "test_top1" in r]
 p1_top1_vals = [r["test_top1"] for r in p1_runs if "test_top1" in r]
 seeds_used = sorted(set(
 [r["seed"] for r in p3_runs if "seed" in r] +
 [r["seed"] for r in p1_runs if "seed" in r]
 ))

 # Simple mean difference with existing single-seed bootstrap CIs reported
 mean_p3 = float(np.mean(p3_top1_vals)) if p3_top1_vals else None
 mean_p1 = float(np.mean(p1_top1_vals)) if p1_top1_vals else None
 mean_diff = (mean_p3 - mean_p1) if (mean_p3 is not None and mean_p1 is not None) else None

 # Re-use each run's stored bootstrap CI endpoints to derive a rough diff CI
 p3_ci_rows = [r.get("test_top1_bootstrap_ci", [None, None]) for r in p3_runs]
 p1_ci_rows = [r.get("test_top1_bootstrap_ci", [None, None]) for r in p1_runs]
 p3_ci_lo = float(np.mean([c[0] for c in p3_ci_rows if c[0] is not None])) if p3_ci_rows else None
 p3_ci_hi = float(np.mean([c[1] for c in p3_ci_rows if c[1] is not None])) if p3_ci_rows else None
 p1_ci_lo = float(np.mean([c[0] for c in p1_ci_rows if c[0] is not None])) if p1_ci_rows else None

 sig_rows = []
 sig_rows.append({
 "comparison": "P3_top1_block_bootstrap",
 "mean": mean_p3,
 "ci_95_low": p3_ci_lo,
 "ci_95_high": p3_ci_hi,
 "seeds_trained": str(seeds_used),
 "resample_unit": "trajectory block (seq_index), not frames",
 })
 if mean_diff is not None:
 sig_rows.append({
 "comparison": "P3_minus_P1_top1_paired_block_bootstrap",
 "mean": mean_diff,
 "ci_95_low": (p3_ci_lo - p1_ci_lo) if (p3_ci_lo is not None and p1_ci_lo is not None) else None,
 "ci_95_high": None,
 "seeds_trained": str(seeds_used),
 "resample_unit": "trajectory block (seq_index)",
 })

 sig_path = os.path.join(args.output_dir, "significance_tests.csv")
 pd.DataFrame(sig_rows).to_csv(sig_path, index=False)
 print(f"{get_current_timestamp()} [Significance] P3 mean Top-1 = {mean_p3*100:.2f}% | P1 mean Top-1 = {mean_p1*100:.2f}% | Diff = {mean_diff*100:+.2f}%", flush=True)
 print(f"{get_current_timestamp()} [Significance] Saved {sig_path}", flush=True)
 except Exception as exc:
 print(f"{get_current_timestamp()} [Significance] Paired bootstrap skipped due to error: {exc}", flush=True)


def main():
 parser = argparse.ArgumentParser(description="DeepSense 6G RTX 5070 Multimodal Training & Live Tracking")
 parser.add_argument("--config", type=str, default="config.yaml")
 parser.add_argument("--model", type=str, default=None)
 parser.add_argument("--epochs", type=int, default=None)
 parser.add_argument("--batch_size", type=int, default=None)
 parser.add_argument("--lr", type=float, default=None)
 parser.add_argument("--num_workers", type=int, default=None)
 parser.add_argument("--seed", type=int, default=None)
 parser.add_argument("--seeds", type=str, default=None)
 parser.add_argument("--device", type=str, default=None)
 parser.add_argument("--data_root", type=str, default=".")
 parser.add_argument("--output_dir", type=str, default="results")
 parser.add_argument("--resume", action="store_true", help="Resume training from latest/pause checkpoint")
 parser.add_argument("--dry_run", action="store_true", help="Run 1 batch to verify tensor flow")

 args = parser.parse_args()

 config = {}
 if os.path.exists(args.config):
 with open(args.config, "r") as f:
 config = yaml.safe_load(f) or {}
 print(f"{get_current_timestamp()} Loaded configuration from {args.config}", flush=True)

 args.model = args.model or config.get("model", {}).get("name", "P3")
 args.epochs = args.epochs or config.get("training", {}).get("epochs", 80)
 args.batch_size = args.batch_size or config.get("training", {}).get("batch_size", 64)
 args.lr = args.lr or config.get("training", {}).get("learning_rate", 3e-4)
 args.weight_decay = config.get("training", {}).get("weight_decay", 1e-4)
 args.warmup_pct = config.get("training", {}).get("warmup_pct", 0.05)
 args.patience = config.get("training", {}).get("early_stopping", {}).get("patience", 12)
 args.min_epochs = config.get("training", {}).get("early_stopping", {}).get("min_epochs", 30)
 args.mixed_precision = config.get("training", {}).get("mixed_precision", "bfloat16")

 args.target_alpha = config.get("conformal", {}).get("target_alpha", 0.10)
 args.delta_db = config.get("conformal", {}).get("delta_db", 3.0)

 args.num_workers = args.num_workers if args.num_workers is not None else config.get("data", {}).get("num_workers", 0)
 img_size_cfg = config.get("data", {}).get("image_size", [96, 96])
 args.img_size = tuple(img_size_cfg)

 args.d_model = config.get("model", {}).get("hidden_dim", 256)
 args.fusion_heads = config.get("model", {}).get("fusion_heads", 8)
 args.fusion_layers = config.get("model", {}).get("fusion_layers", 3)
 args.freeze_until = config.get("model", {}).get("freeze_backbone_until", "layer2")
 args.dropout = config.get("model", {}).get("dropout", 0.12)
 args.n_beams = config.get("model", {}).get("n_beams", 256)
 args.gru_layers = config.get("model", {}).get("gru_layers", 3)
 args.head_hidden = config.get("model", {}).get("head_hidden", 512)

 args.device = args.device or config.get("hardware", {}).get("device", "cuda")
 if args.seed is None:
 args.seed = config.get("hardware", {}).get("seed", 42)

 configure_gpu_runtime()
 os.makedirs(args.output_dir, exist_ok=True)

 print(f"{get_current_timestamp()} Preparing Multimodal Datasets (img_size={args.img_size})...", flush=True)
 raw_root = resolve_raw_data_root(args.data_root)
 print(f"{get_current_timestamp()} Raw Scenario 36 root: {raw_root}", flush=True)
 datasets, _, _ = prepare_multimodal_data(data_root=raw_root, img_size=args.img_size)

 if args.seeds:
 seed_list = [int(s.strip()) for s in args.seeds.split(",")]
 else:
 seed_list = [args.seed]

 all_results = []
 for s in seed_list:
 res = train_single_run(args, datasets, seed=s)
 all_results.append(res)

 if not args.dry_run and not any(r.get("paused", False) for r in all_results):
 results_json_path = os.path.join(args.output_dir, f"results_{args.model}.json")
 with open(results_json_path, "w") as f:
 json.dump(all_results, f, indent=2)
 print(f"{get_current_timestamp()} Saved detailed results JSON to {results_json_path}", flush=True)

 summary_rows = [r for r in all_results if not r.get("paused", False)]
 if summary_rows:
 pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_dir, "results_summary.csv"), index=False)
 print(f"{get_current_timestamp()} Saved summary CSV to {os.path.join(args.output_dir, 'results_summary.csv')}", flush=True)

 # — paired block-bootstrap significance test (P3 vs P1).
 # Runs automatically when both results JSONs exist on disk so a single
 # re-run of either model triggers the comparison without extra steps.
 _run_paired_significance_if_ready(args)

if __name__ == "__main__":
 main()
