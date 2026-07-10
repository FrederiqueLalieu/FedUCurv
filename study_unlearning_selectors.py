import argparse
import json
import math
import pickle
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from MIA import (
    build_client_feature_rows,
    score_client_deletion_attack,
    train_client_deletion_attack,
)
from data_utils import load_pkl_as_dataset, make_federated_loaders, make_global_loader_from_clients
from federated import simulate_federated_learning
from training import Net, evaluate
from unlearning import flatten_state_dict, predictive_metrics, unlearn


OUTLIER_CLIENTS = [0, 3, 4, 6, 8, 9]
ETA_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]
DAMPING_OFFSETS = [-0.05, -0.025, 0.0, 0.025, 0.05]
LOCAL_OUTPUT_SCALES = [1.0, 1.25, 1.5]
HIDDEN_LAYER_SCALES = [0.5, 0.75, 1.0]
RECENT_SUMMARY_WINDOWS = [1, 2, 3, 4, 5]


def progress(message):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_float_grid(text):
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--artificial", default=False)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eta-grid", default="0.0015,0.002,0.0025,0.003,0.0035,0.004,0.005,0.006,0.008,0.01")
    parser.add_argument("--damping-grid", default="0.05,0.075,0.1,0.125,0.15,0.2")
    parser.add_argument("--floor-ratios", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--scale-factors", default="0.4,0.6,0.8,1.0,1.2,1.5,2.0")
    parser.add_argument("--loss-band-eps", default="0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02")
    parser.add_argument("--guarded-auc-tols", default="0.001,0.002,0.003,0.005,0.0075,0.01,0.015")
    parser.add_argument("--guarded-ece-tols", default="0.002,0.003,0.005,0.0075,0.01,0.015,0.02")
    parser.add_argument("--score-loss-weights", default="0.5,1.0,1.5,2.0,3.0")
    parser.add_argument("--score-auc-weights", default="0.25,0.5,1.0,1.5,2.0")
    parser.add_argument("--score-ece-weights", default="0.25,0.5,1.0,1.5,2.0")
    parser.add_argument("--score-step-weights", default="0.5,1.0,1.5,2.0,3.0")
    parser.add_argument("--output-scales", default="1.0")
    parser.add_argument("--client-mia-report", default=None)
    parser.add_argument("--mia-auc-weight", type=float, default=2.0)
    parser.add_argument("--mia-gap-weight", type=float, default=1.5)
    parser.add_argument("--mia-rank-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def export_config_artifact(report, path):
    payload = {
        "source_rule": report["name"],
        "selected_configs": [],
    }
    for row in report["selected_configs"]:
        config_row = {
            "client": int(row["client"]),
            "eta": float(row["eta"]),
            "damping": float(row["damping"]),
            "output_layer_scale": float(row.get("output_layer_scale", 1.0)),
        }
        if abs(float(row.get("hidden_layer_scale", 1.0)) - 1.0) > 1e-12:
            config_row["hidden_layer_scale"] = float(row["hidden_layer_scale"])
        if int(row.get("summary_window", 0)) > 0:
            config_row["summary_window"] = int(row["summary_window"])
        payload["selected_configs"].append(config_row)
    if "selector_config" in report:
        payload["selector_config"] = report["selector_config"]
    save_json(payload, path)


def safe_mean(values):
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return float("nan")
    return float(np.mean(clean))


def sort_key_for_oracle(row):
    return (
        row["predictive_kl"],
        row["test_acc_gap_abs"],
        row["test_auc_gap_abs"],
        row["test_f1_gap_abs"],
    )


def sort_key_for_retained_loss(row):
    return (
        row["retained_val_loss"],
        -row["retained_val_auc"],
        row["retained_val_ece"],
    )


def model_from_params(params, input_size):
    model = Net(input_size=input_size)
    model.load_state_dict(params)
    return model


def loader_from_indices(dataset, indices, batch_size):
    subset = Subset(dataset, np.asarray(indices, dtype=int))
    return DataLoader(subset, batch_size=batch_size, shuffle=False)


def scalar_metrics_from_eval(model, loader, device):
    loss, acc, prec, recall, conf, f1, auc, brier, ece, pos_rate, valid_rate, non_finite_rate = evaluate(
        model,
        loader,
        device,
    )
    return {
        "loss": float(loss),
        "acc": float(acc),
        "prec": float(prec),
        "recall": float(recall),
        "confidence": float(conf),
        "f1": float(f1),
        "auc": float(auc),
        "brier": float(brier),
        "ece": float(ece),
        "positive_rate": float(pos_rate),
        "valid_rate": float(valid_rate),
        "non_finite_rate": float(non_finite_rate),
    }


def candidate_key(row):
    return (
        int(row["client"]),
        round(float(row["eta"]), 12),
        round(float(row["damping"]), 12),
        round(float(row.get("output_layer_scale", 1.0)), 12),
        round(float(row.get("hidden_layer_scale", 1.0)), 12),
        int(row.get("summary_window", 0)),
    )


def normalize_candidate_row(row):
    normalized = dict(row)
    normalized["client"] = int(row["client"])
    normalized["eta"] = float(row["eta"])
    normalized["damping"] = float(row["damping"])
    normalized["output_layer_scale"] = float(row.get("output_layer_scale", 1.0))
    normalized["hidden_layer_scale"] = float(row.get("hidden_layer_scale", 1.0))
    normalized["summary_window"] = int(row.get("summary_window", 0))
    return normalized


def aggregate_selected_rows(rows):
    metric_keys = [
        "predictive_kl",
        "predictive_js",
        "predictive_tv",
        "predictive_agreement",
        "predictive_prob_rmse",
        "retained_val_loss",
        "retained_val_auc",
        "retained_val_ece",
        "forgotten_val_loss",
        "forgotten_val_auc",
        "forgotten_val_ece",
        "test_acc_gap_abs",
        "test_auc_gap_abs",
        "test_f1_gap_abs",
        "unlearn_test_acc",
        "unlearn_test_auc",
        "unlearn_test_f1",
        "unlearn_test_brier",
        "unlearn_test_ece",
        "step_l2",
    ]
    optional_metric_keys = [
        "mia_retrain_auc_target",
        "mia_retrain_gap_target",
        "mia_retrain_rank_target",
        "mia_candidate_auc",
        "mia_candidate_gap",
        "mia_candidate_rank",
        "mia_auc_match_abs",
        "mia_gap_match_abs",
        "mia_rank_match_abs",
    ]
    if rows and any(key in rows[0] for key in optional_metric_keys):
        metric_keys.extend([key for key in optional_metric_keys if key in rows[0]])
    return {metric: safe_mean([row[metric] for row in rows]) for metric in metric_keys}


def build_candidate_cache(
    base_dir,
    data_dir,
    output_dir,
    eta_grid,
    damping_grid,
    output_scales,
    batch_size,
    seed
):
    """ Make the federated dataloaders, train the original model, create the 
    retrained models for each client. Then, for each combination of damping 
    factor and eta, perform unlearning. It reports how well the retrained 
    model performed, the performances of the unlearned model on the unlearned and 
    retained cleints separately and their difference metrics. 
    """
    progress("loading saved experiment assets")
    base_dir = Path(base_dir)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, metadata = load_pkl_as_dataset(data_dir / "train_data.pkl")
    test_dataset, _ = load_pkl_as_dataset(data_dir / "test_data.pkl")
    input_size = metadata["input_size"]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    partitions = pickle.load(open(base_dir / "partitions.pkl", "rb"))
    client_train_idx = partitions["client_train_idx"]
    client_val_idx = partitions["client_val_idx"]
    _, client_val_loaders = make_federated_loaders(
        train_dataset,
        client_train_idx,
        client_val_idx,
        batch_size=batch_size,
        seed=seed,
    )

    retained_val_loaders = []
    forgotten_val_loaders = []
    for client in range(len(client_val_idx)):
        retained_idx = client_val_idx[:client] + client_val_idx[client + 1:]
        retained_loader, _ = make_global_loader_from_clients(
            train_dataset,
            retained_idx,
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
        )
        retained_val_loaders.append(retained_loader)
        forgotten_val_loaders.append(loader_from_indices(train_dataset, client_val_idx[client], batch_size))

    og_params = torch.load(base_dir / "og_params.pt", map_location="cpu", weights_only=True)
    og_hessian = torch.load(base_dir / "og_hessian", map_location="cpu", weights_only=False)
    og_summaries = torch.load(base_dir / "og_summaries", map_location="cpu", weights_only=False)
    summary_bank = {0: og_summaries}
    og_params_vec = flatten_state_dict(og_params).float()
    og_hessian_vec = flatten_state_dict(og_hessian).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress(f"using device {device}")

    candidate_rows = {}
    total_candidates = len(client_val_idx) * len(eta_grid) * len(damping_grid) * len(output_scales)
    done = 0

    for client in range(len(client_val_idx)):
        progress(f"building cache for client {client}")
        retr_params = torch.load(base_dir / f"{client}" / "retr_params.pt", map_location="cpu", weights_only=True)
        retr_test_results = pickle.load(open(base_dir / f"{client}" / "retr_test_results", "rb"))
        summary_vec = flatten_state_dict(og_summaries[client]).float()
        summary_l2 = float(summary_vec.norm().item())
        summary_l1 = float(summary_vec.abs().sum().item())
        client_rows = []

        for damping in damping_grid:
            safe_hessian = og_hessian_vec + float(damping)
            safe_hessian = torch.clamp(safe_hessian, min=1e-8)
            raw_step_vec = summary_vec / safe_hessian
            raw_step_l2 = float(raw_step_vec.norm().item())
            raw_step_l1 = float(raw_step_vec.abs().sum().item())

            for eta in eta_grid:
                for output_scale in output_scales:
                    unl_params = unlearn(
                        og_params,
                        og_hessian,
                        og_summaries,
                        client,
                        len(client_val_idx),
                        eta,
                        damping,
                        step_sign=1.0,
                        output_layer_scale=output_scale,
                        hidden_layer_scale=1.0,
                    )
                    step_vec = flatten_state_dict(unl_params).float() - og_params_vec
                    step_l2 = float(step_vec.norm().item())
                    step_l1 = float(step_vec.abs().sum().item())

                    unl_model = model_from_params(unl_params, input_size)
                    unl_test = scalar_metrics_from_eval(unl_model, test_loader, device)
                    retained_val = scalar_metrics_from_eval(unl_model, retained_val_loaders[client], device)
                    forgotten_val = scalar_metrics_from_eval(unl_model, forgotten_val_loaders[client], device)
                    predictive = predictive_metrics(retr_params, unl_params, test_loader, device)

                    client_rows.append(
                        {
                            "client": client,
                            "eta": float(eta),
                            "damping": float(damping),
                            "output_layer_scale": float(output_scale),
                            "hidden_layer_scale": 1.0,
                            "summary_window": 0,
                            "summary_l2": summary_l2,
                            "summary_l1": summary_l1,
                            "raw_step_l2": raw_step_l2,
                            "raw_step_l1": raw_step_l1,
                            "step_l2": step_l2,
                            "step_l1": step_l1,
                            "predictive_kl": float(predictive["kl"]),
                            "predictive_js": float(predictive["js"]),
                            "predictive_tv": float(predictive["tv"]),
                            "predictive_agreement": float(predictive["agreement"]),
                            "predictive_prob_rmse": float(predictive["prob_rmse"]),
                            "retained_val_loss": retained_val["loss"],
                            "retained_val_auc": retained_val["auc"],
                            "retained_val_ece": retained_val["ece"],
                            "forgotten_val_loss": forgotten_val["loss"],
                            "forgotten_val_auc": forgotten_val["auc"],
                            "forgotten_val_ece": forgotten_val["ece"],
                            "unlearn_test_acc": unl_test["acc"],
                            "unlearn_test_auc": unl_test["auc"],
                            "unlearn_test_f1": unl_test["f1"],
                            "unlearn_test_brier": unl_test["brier"],
                            "unlearn_test_ece": unl_test["ece"],
                            "test_acc_gap_abs": abs(unl_test["acc"] - float(retr_test_results["acc"])),
                            "test_auc_gap_abs": abs(unl_test["auc"] - float(retr_test_results["auc"])),
                            "test_f1_gap_abs": abs(unl_test["f1"] - float(retr_test_results["f1"])),
                        }
                    )
                    done += 1

            progress(f"client {client} cached damping {damping:.3f} ({done}/{total_candidates})")

        candidate_rows[str(client)] = client_rows

    cache = {
        "config": {
            "base_dir": str(base_dir),
            "data_dir": str(data_dir),
            "eta_grid": eta_grid,
            "damping_grid": damping_grid,
            "output_scales": output_scales,
            "batch_size": batch_size,
            "seed": seed,
        },
        "candidate_rows": candidate_rows,
    }
    save_json(cache, output_dir / "selector_cache.json")
    return cache


def load_client_mia_report(path):
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as handle:
        obj = json.load(handle)

    if isinstance(obj, dict) and "aggregate" in obj and "per_client" in obj:
        return obj

    if isinstance(obj, dict):
        nested = [value for value in obj.values() if isinstance(value, dict) and "aggregate" in value and "per_client" in value]
        if len(nested) == 1:
            return nested[0]

    raise ValueError(f"could not find a client mia report in {path}")


def load_external_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        obj = json.load(handle)

    rows = obj["selected_configs"] if isinstance(obj, dict) and "selected_configs" in obj else obj
    configs = {}
    for row in rows:
        configs[int(row["client"])] = {
            "eta": float(row["eta"]),
            "damping": float(row["damping"]),
            "output_layer_scale": float(row.get("output_layer_scale", 1.0)),
            "hidden_layer_scale": float(row.get("hidden_layer_scale", 1.0)),
            "summary_window": int(row.get("summary_window", 0)),
        }
    return configs


def sum_client_deltas(delta_rounds):
    num_clients = len(delta_rounds[0])
    summaries = []
    for client in range(num_clients):
        client_summary = {
            key: torch.zeros_like(value)
            for key, value in delta_rounds[0][client].items()
        }
        for round_rows in delta_rounds:
            for key, value in round_rows[client].items():
                client_summary[key] = client_summary[key] + value
        summaries.append(client_summary)
    return summaries


def build_recent_summary_bank(client_delta_rounds):
    summary_bank = {0: sum_client_deltas(client_delta_rounds)}
    for window in RECENT_SUMMARY_WINDOWS:
        clipped = min(int(window), len(client_delta_rounds))
        summary_bank[clipped] = sum_client_deltas(client_delta_rounds[-clipped:])
    return summary_bank


def ensure_original_round_deltas(base_dir, data_dir, batch_size, seed):
    base_dir = Path(base_dir)
    rounds_path = base_dir / "og_client_delta_rounds.pt"
    if rounds_path.exists():
        return torch.load(rounds_path, map_location="cpu", weights_only=False)

    progress("rebuilding original round deltas for recent-summary study")
    with open(base_dir / "experiment_summary.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)["config"]

    set_seed(int(config["seed"]))
    train_dataset, _ = load_pkl_as_dataset(Path(data_dir) / "train_data.pkl")
    partitions = pickle.load(open(base_dir / "partitions.pkl", "rb"))
    client_train_loaders, client_val_loaders = make_federated_loaders(
        train_dataset,
        partitions["client_train_idx"],
        partitions["client_val_idx"],
        batch_size=int(config["batch_size"]),
        seed=int(config["seed"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        _model,
        _summaries,
        _curvature,
        _results,
        _trajectory,
        _summary_trajectory,
        _convergence_metrics,
        client_delta_rounds,
    ) = simulate_federated_learning(
        client_train_loaders,
        client_val_loaders,
        device,
        epochs=int(config["local_epochs"]),
        lr=float(config["fl_lr"]),
        batch_size=int(config["batch_size"]),
        alpha=float(config["alpha"]),
        max_rounds=int(config["fl_rounds"]),
        run_label="original history rebuild",
        progress=True,
    )
    torch.save(client_delta_rounds, rounds_path)
    return client_delta_rounds


def build_mia_context(base_dir, data_dir, batch_size, report):
    """ Loads the clients, original model and retrained models. Then for 
    different combinations of seeds and the variable c (regularization 
    parameter, higher c is more fitting to data, lower is more regularized), 
    train an attack model for each client. For each MIA model, get auc.
    """
    progress("loading client mia context")
    train_dataset, metadata = load_pkl_as_dataset(Path(data_dir) / "train_data.pkl")
    input_size = metadata["input_size"]
    partitions = pickle.load(open(Path(base_dir) / "partitions.pkl", "rb"))
    client_train_idx = partitions["client_train_idx"]
    client_loaders = [
        loader_from_indices(train_dataset, indices, batch_size=max(batch_size, int(report["config"]["bag_size"])))
        for indices in client_train_idx
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    og_params = torch.load(Path(base_dir) / "og_params.pt", map_location="cpu", weights_only=True)
    original_model = model_from_params(og_params, input_size)
    original_rows = build_client_feature_rows(original_model, client_loaders, device)

    retrain_rows = []
    for removed_client in range(len(client_loaders)):
        retrain_params = torch.load(
            Path(base_dir) / f"{removed_client}" / "retr_params.pt",
            map_location="cpu",
            weights_only=True,
        )
        retrain_model = model_from_params(retrain_params, input_size)
        retrain_rows.append(build_client_feature_rows(retrain_model, client_loaders, device))

    selected_attack_rows = report["config"].get("selected_attack_c_by_seed", [])
    if not selected_attack_rows and "per_seed" in report:
        for seed_key, seed_row in report["per_seed"].items():
            seed = int(seed_key.split("_")[-1])
            selected_attack_rows.append(
                {
                    "seed": seed,
                    "attack_c": float(seed_row["selected_attack_c"]),
                }
            )

    seed_runs = []
    for attack_info in selected_attack_rows:
        seed = int(attack_info["seed"])
        attack_c = float(attack_info["attack_c"])
        progress(f"training client mia attacks for seed {seed} with c={attack_c}")
        attacks = {}
        for holdout_client in range(len(client_loaders)):
            attack_model, _ = train_client_deletion_attack(
                original_rows,
                retrain_rows,
                holdout_client,
                train_bags=int(report["config"]["train_bags"]),
                cross_negative_bags=int(report["config"]["cross_negative_bags"]),
                bag_size=int(report["config"]["bag_size"]),
                attack_c=attack_c,
                seed=seed,
            )
            attacks[holdout_client] = attack_model
        seed_runs.append(
            {
                "seed": seed,
                "attack_c": attack_c,
                "attacks": attacks,
            }
        )

    targets = {}
    gap_metric = str(report.get("config", {}).get("gap_metric", "score_gap"))
    auc_metric = str(report.get("config", {}).get("auc_metric", "bag_auc"))
    auc_field = "auc" if auc_metric == "bag_auc" else auc_metric
    for client_key, row in report["per_client"].items():
        retrain_row = row["retrain"]
        targets[int(client_key)] = {
            "mia_retrain_auc_target": float(retrain_row.get(auc_field, retrain_row["auc"])),
            "mia_retrain_gap_target": float(retrain_row.get(gap_metric, retrain_row["score_gap"])),
            "mia_retrain_rank_target": float(retrain_row["rank"]),
        }

    return {
        "device": device,
        "input_size": input_size,
        "client_loaders": client_loaders,
        "original_rows": original_rows,
        "seed_runs": seed_runs,
        "targets": targets,
        "eval_bags": int(report["config"]["eval_bags"]),
        "bag_size": int(report["config"]["bag_size"]),
        "auc_field": auc_field,
        "gap_metric": gap_metric,
    }


def build_candidate_evaluator(base_dir, data_dir, batch_size, seed, client_mia_report=None):
    """ Create federated loaders, load the original model from storage, load the
    retrained models from storage, build the MIA context. 
    """
    base_dir = Path(base_dir)
    data_dir = Path(data_dir)
    train_dataset, metadata = load_pkl_as_dataset(data_dir / "train_data.pkl")
    test_dataset, _ = load_pkl_as_dataset(data_dir / "test_data.pkl")
    input_size = metadata["input_size"]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    partitions = pickle.load(open(base_dir / "partitions.pkl", "rb"))
    client_train_idx = partitions["client_train_idx"]
    client_val_idx = partitions["client_val_idx"]
    _, client_val_loaders = make_federated_loaders(
        train_dataset,
        client_train_idx,
        client_val_idx,
        batch_size=batch_size,
        seed=seed,
    )
    retained_val_loaders = []
    forgotten_val_loaders = []
    for client in range(len(client_val_idx)):
        retained_idx = client_val_idx[:client] + client_val_idx[client + 1:]
        retained_loader, _ = make_global_loader_from_clients(
            train_dataset,
            retained_idx,
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
        )
        retained_val_loaders.append(retained_loader)
        forgotten_val_loaders.append(loader_from_indices(train_dataset, client_val_idx[client], batch_size))

    og_params = torch.load(base_dir / "og_params.pt", map_location="cpu", weights_only=True)
    og_hessian = torch.load(base_dir / "og_hessian", map_location="cpu", weights_only=False)
    og_summaries = torch.load(base_dir / "og_summaries", map_location="cpu", weights_only=False)
    summary_bank = {0: og_summaries}
    og_params_vec = flatten_state_dict(og_params).float()
    og_hessian_vec = flatten_state_dict(og_hessian).float()
    retr_params_by_client = [
        torch.load(base_dir / f"{client}" / "retr_params.pt", map_location="cpu", weights_only=True)
        for client in range(len(client_val_idx))
    ]
    retr_test_results = [
        pickle.load(open(base_dir / f"{client}" / "retr_test_results", "rb"))
        for client in range(len(client_val_idx))
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mia_context = None
    if client_mia_report is not None:
        mia_context = build_mia_context(base_dir, data_dir, batch_size, client_mia_report)

    feature_row_cache = {}

    params_cache = {}
    model_cache = {}

    def get_summaries_for_window(summary_window): 
        summary_window = int(summary_window)
        if summary_window in summary_bank:
            return summary_bank[summary_window]
        client_delta_rounds = ensure_original_round_deltas(base_dir, data_dir, batch_size, seed)
        summary_bank.update(build_recent_summary_bank(client_delta_rounds))
        return summary_bank.get(summary_window, summary_bank[0])

    def build_unlearned_model(
        client,
        eta,
        damping,
        output_layer_scale=1.0,
        hidden_layer_scale=1.0,
        summary_window=0,
    ):
        key = candidate_key(
            {
                "client": int(client),
                "eta": float(eta),
                "damping": float(damping),
                "output_layer_scale": float(output_layer_scale),
                "hidden_layer_scale": float(hidden_layer_scale),
                "summary_window": int(summary_window),
            }
        )
        selected_summaries = get_summaries_for_window(summary_window)
        if key not in params_cache:
            params_cache[key] = unlearn(
                og_params,
                og_hessian,
                selected_summaries,
                client,
                len(client_val_idx),
                eta,
                damping,
                step_sign=1.0,
                output_layer_scale=output_layer_scale,
                hidden_layer_scale=hidden_layer_scale,
            )
        if key not in model_cache:
            model_cache[key] = model_from_params(params_cache[key], input_size)
        return key, params_cache[key], model_cache[key]

    def attach_mia_metrics(row):
        if mia_context is None or "mia_candidate_auc" in row:
            return row

        client = int(row["client"])
        eta = float(row["eta"])
        damping = float(row["damping"])
        output_layer_scale = float(row.get("output_layer_scale", 1.0))
        hidden_layer_scale = float(row.get("hidden_layer_scale", 1.0))
        summary_window = int(row.get("summary_window", 0))
        key, _, unl_model = build_unlearned_model(
            client,
            eta,
            damping,
            output_layer_scale=output_layer_scale,
            hidden_layer_scale=hidden_layer_scale,
            summary_window=summary_window,
        )
        if key not in feature_row_cache:
            feature_row_cache[key] = build_client_feature_rows(unl_model, mia_context["client_loaders"], mia_context["device"])
        candidate_rows = feature_row_cache[key]
        seed_scores = []
        for seed_run in mia_context["seed_runs"]:
            score = score_client_deletion_attack(
                seed_run["attacks"][client],
                mia_context["original_rows"],
                candidate_rows,
                client,
                eval_bags=mia_context["eval_bags"],
                bag_size=mia_context["bag_size"],
                seed=int(seed_run["seed"]),
            )
            seed_scores.append(score)

        updated = dict(row)
        updated["mia_candidate_auc"] = float(
            np.mean([score.get(mia_context["auc_field"], score["auc"]) for score in seed_scores])
        )
        updated["mia_candidate_gap"] = float(
            np.mean([score.get(mia_context["gap_metric"], score["score_gap"]) for score in seed_scores])
        )
        updated["mia_candidate_rank"] = float(np.mean([score["rank"] for score in seed_scores]))
        updated.update(mia_context["targets"][client])
        updated["mia_auc_match_abs"] = abs(updated["mia_candidate_auc"] - updated["mia_retrain_auc_target"])
        updated["mia_gap_match_abs"] = abs(updated["mia_candidate_gap"] - updated["mia_retrain_gap_target"])
        updated["mia_rank_match_abs"] = abs(updated["mia_candidate_rank"] - updated["mia_retrain_rank_target"])
        return updated

    def evaluate_candidate_row(
        client,
        eta,
        damping,
        output_layer_scale=1.0,
        hidden_layer_scale=1.0,
        summary_window=0,
    ):
        row = {
            "client": int(client),
            "eta": float(eta),
            "damping": float(damping),
            "output_layer_scale": float(output_layer_scale),
            "hidden_layer_scale": float(hidden_layer_scale),
            "summary_window": int(summary_window),
        }
        key = candidate_key(row)

        selected_summaries = get_summaries_for_window(summary_window)
        summary_vec = flatten_state_dict(selected_summaries[client]).float()
        summary_l2 = float(summary_vec.norm().item())
        summary_l1 = float(summary_vec.abs().sum().item())
        safe_hessian = og_hessian_vec + float(damping)
        safe_hessian = torch.clamp(safe_hessian, min=1e-8)
        raw_step_vec = summary_vec / safe_hessian
        raw_step_l2 = float(raw_step_vec.norm().item())
        raw_step_l1 = float(raw_step_vec.abs().sum().item())

        key, unl_params, unl_model = build_unlearned_model(
            client,
            eta,
            damping,
            output_layer_scale=output_layer_scale,
            hidden_layer_scale=hidden_layer_scale,
            summary_window=summary_window,
        )
        step_vec = flatten_state_dict(unl_params).float() - og_params_vec
        step_l2 = float(step_vec.norm().item())
        step_l1 = float(step_vec.abs().sum().item())
        unl_test = scalar_metrics_from_eval(unl_model, test_loader, device)
        retained_val = scalar_metrics_from_eval(unl_model, retained_val_loaders[client], device)
        forgotten_val = scalar_metrics_from_eval(unl_model, forgotten_val_loaders[client], device)
        predictive = predictive_metrics(retr_params_by_client[client], unl_params, test_loader, device)

        row.update(
            {
                "summary_l2": summary_l2,
                "summary_l1": summary_l1,
                "raw_step_l2": raw_step_l2,
                "raw_step_l1": raw_step_l1,
                "step_l2": step_l2,
                "step_l1": step_l1,
                "predictive_kl": float(predictive["kl"]),
                "predictive_js": float(predictive["js"]),
                "predictive_tv": float(predictive["tv"]),
                "predictive_agreement": float(predictive["agreement"]),
                "predictive_prob_rmse": float(predictive["prob_rmse"]),
                "retained_val_loss": retained_val["loss"],
                "retained_val_auc": retained_val["auc"],
                "retained_val_ece": retained_val["ece"],
                "forgotten_val_loss": forgotten_val["loss"],
                "forgotten_val_auc": forgotten_val["auc"],
                "forgotten_val_ece": forgotten_val["ece"],
                "unlearn_test_acc": unl_test["acc"],
                "unlearn_test_auc": unl_test["auc"],
                "unlearn_test_f1": unl_test["f1"],
                "unlearn_test_brier": unl_test["brier"],
                "unlearn_test_ece": unl_test["ece"],
                "test_acc_gap_abs": abs(unl_test["acc"] - float(retr_test_results[client]["acc"])),
                "test_auc_gap_abs": abs(unl_test["auc"] - float(retr_test_results[client]["auc"])),
                "test_f1_gap_abs": abs(unl_test["f1"] - float(retr_test_results[client]["f1"])),
            }
        )

        return attach_mia_metrics(row)

    return evaluate_candidate_row, attach_mia_metrics


def materialize_config_rows(rows_by_client, config_by_client, evaluate_candidate_row):
    selected = []
    existing_by_client = {
        client: {candidate_key(row): row for row in rows}
        for client, rows in rows_by_client.items()
    }

    for client, config in sorted(config_by_client.items()):
        key = candidate_key(
            {
                "client": client,
                "eta": config["eta"],
                "damping": config["damping"],
                "output_layer_scale": config.get("output_layer_scale", 1.0),
                "hidden_layer_scale": config.get("hidden_layer_scale", 1.0),
                "summary_window": config.get("summary_window", 0),
            }
        )
        row = existing_by_client[client].get(key)
        if row is None:
            row = evaluate_candidate_row(
                client,
                config["eta"],
                config["damping"],
                output_layer_scale=config.get("output_layer_scale", 1.0),
                hidden_layer_scale=config.get("hidden_layer_scale", 1.0),
                summary_window=config.get("summary_window", 0),
            )
            rows_by_client[client].append(row)
            existing_by_client[client][key] = row
        selected.append(row)
    return selected


def select_baseline(rows_by_client, eta, damping, output_layer_scale=1.0):
    """ filters per-client results down to the unique row that corresponds 
    to a specific baseline hyperparameter setting.
    """
    selected = []
    for rows in rows_by_client.values():
        match = next(
            row
            for row in rows
            if abs(row["eta"] - eta) < 1e-12
            and abs(row["damping"] - damping) < 1e-12
            and abs(row.get("output_layer_scale", 1.0) - output_layer_scale) < 1e-12
            and abs(row.get("hidden_layer_scale", 1.0) - 1.0) < 1e-12
            and int(row.get("summary_window", 0)) == 0
        )
        selected.append(match)
    return selected


def select_oracle(rows_by_client):
    return [min(rows, key=sort_key_for_oracle) for rows in rows_by_client.values()]


def select_retained_loss(rows_by_client):
    return [min(rows, key=sort_key_for_retained_loss) for rows in rows_by_client.values()]


def select_retained_loss_with_floor(rows_by_client, baseline_rows, floor_ratio):
    selected = []
    for rows, baseline_row in zip(rows_by_client.values(), baseline_rows):
        floor_value = floor_ratio * baseline_row["step_l2"]
        kept = [row for row in rows if row["step_l2"] >= floor_value]
        if not kept:
            kept = rows
        selected.append(min(kept, key=sort_key_for_retained_loss))
    return selected


def select_step_norm_rule(rows_by_client, damping, target_step_l2):
    selected = []
    for rows in rows_by_client.values():
        kept = [row for row in rows if abs(row["damping"] - damping) < 1e-12]
        selected.append(
            min(
                kept,
                key=lambda row: (
                    abs(row["step_l2"] - target_step_l2),
                    row["retained_val_loss"],
                ),
            )
        )
    return selected


def select_summary_norm_rule(rows_by_client, damping, scale):
    selected = []
    for rows in rows_by_client.values():
        kept = [row for row in rows if abs(row["damping"] - damping) < 1e-12]
        desired_eta = scale / max(kept[0]["summary_l2"], 1e-8)
        selected.append(
            min(
                kept,
                key=lambda row: (
                    abs(row["eta"] - desired_eta),
                    row["retained_val_loss"],
                ),
            )
        )
    return selected


def select_loss_band_largest_step(rows_by_client, loss_eps):
    selected = []
    for rows in rows_by_client.values():
        min_loss = min(row["retained_val_loss"] for row in rows)
        kept = [row for row in rows if row["retained_val_loss"] <= min_loss + loss_eps]
        selected.append(
            max(
                kept,
                key=lambda row: (
                    row["step_l2"],
                    row["eta"],
                    -row["retained_val_loss"],
                ),
            )
        )
    return selected


def select_guarded_loss_band(rows_by_client, loss_eps, auc_tol, ece_tol):
    selected = []
    for rows in rows_by_client.values():
        min_loss = min(row["retained_val_loss"] for row in rows)
        max_auc = max(row["retained_val_auc"] for row in rows)
        min_ece = min(row["retained_val_ece"] for row in rows)
        kept = [
            row
            for row in rows
            if row["retained_val_loss"] <= min_loss + loss_eps
            and row["retained_val_auc"] >= max_auc - auc_tol
            and row["retained_val_ece"] <= min_ece + ece_tol
        ]
        if not kept:
            kept = [row for row in rows if row["retained_val_loss"] <= min_loss + loss_eps]
        if not kept:
            kept = rows
        selected.append(
            max(
                kept,
                key=lambda row: (
                    row["step_l2"],
                    row["eta"],
                    -row["retained_val_loss"],
                ),
            )
        )
    return selected


def normalized_values(rows, key, invert=False):
    values = [(-row[key] if invert else row[key]) for row in rows]
    low = min(values)
    high = max(values)
    span = high - low
    if span < 1e-12:
        return [0.0 for _ in values]
    return [(value - low) / span for value in values]


def select_balanced_score_rule(rows_by_client, loss_weight, auc_weight, ece_weight, step_weight):
    selected = []
    for rows in rows_by_client.values():
        loss_scores = normalized_values(rows, "retained_val_loss")
        auc_scores = normalized_values(rows, "retained_val_auc", invert=True)
        ece_scores = normalized_values(rows, "retained_val_ece")
        step_scores = normalized_values(rows, "step_l2", invert=True)

        candidates = []
        for row, loss_score, auc_score, ece_score, step_score in zip(
            rows,
            loss_scores,
            auc_scores,
            ece_scores,
            step_scores,
        ):
            score = (
                loss_weight * loss_score
                + auc_weight * auc_score
                + ece_weight * ece_score
                + step_weight * step_score
            )
            candidates.append((score, row))
        selected.append(min(candidates, key=lambda item: (item[0], item[1]["retained_val_loss"]))[1])
    return selected


def passes_retrain_match_guardrails(row):
    if row["predictive_kl"] > 0.08:
        return False
    if row["test_acc_gap_abs"] > 0.045:
        return False
    if row["test_auc_gap_abs"] > 0.04:
        return False
    if row["test_f1_gap_abs"] > 0.13:
        return False
    if row["mia_retrain_auc_target"] >= 0.60 and row["mia_candidate_auc"] < 0.50:
        return False
    return True


def select_retrain_match_rule(rows_by_client, mia_auc_weight, mia_gap_weight, mia_rank_weight):
    selected = []
    for rows in rows_by_client.values():
        kept = [row for row in rows if passes_retrain_match_guardrails(row)]
        if not kept:
            kept = rows

        auc_scores = normalized_values(kept, "mia_auc_match_abs")
        gap_scores = normalized_values(kept, "mia_gap_match_abs")
        rank_scores = normalized_values(kept, "mia_rank_match_abs")
        kl_scores = normalized_values(kept, "predictive_kl")
        f1_scores = normalized_values(kept, "test_f1_gap_abs")
        acc_scores = normalized_values(kept, "test_acc_gap_abs")
        auc_gap_scores = normalized_values(kept, "test_auc_gap_abs")
        loss_scores = normalized_values(kept, "retained_val_loss")

        candidates = []
        for row, auc_score, gap_score, rank_score, kl_score, f1_score, acc_score, auc_gap_score, loss_score in zip(
            kept,
            auc_scores,
            gap_scores,
            rank_scores,
            kl_scores,
            f1_scores,
            acc_scores,
            auc_gap_scores,
            loss_scores,
        ):
            score = (
                mia_auc_weight * auc_score
                + mia_gap_weight * gap_score
                + mia_rank_weight * rank_score
                + 1.0 * kl_score
                + 0.75 * f1_score
                + 0.5 * acc_score
                + 0.5 * auc_gap_score
                + 0.25 * loss_score
            )
            candidates.append((score, row))
        selected.append(
            min(
                candidates,
                key=lambda item: (
                    item[0],
                    item[1]["mia_auc_match_abs"],
                    item[1]["predictive_kl"],
                ),
            )[1]
        )
    return selected


def passes_constrained_retrain_match(row):
    return (
        row["predictive_kl"] <= 0.05
        and row["test_acc_gap_abs"] <= 0.02
        and row["test_auc_gap_abs"] <= 0.02
        and row["test_f1_gap_abs"] <= 0.05
        and row["mia_candidate_auc"] >= 0.50
    )


def select_constrained_retrain_match_rule(rows_by_client):
    selected = []
    for rows in rows_by_client.values():
        feasible = [row for row in rows if passes_constrained_retrain_match(row)]
        if not feasible:
            feasible = rows
        selected.append(
            min(
                feasible,
                key=lambda row: (
                    row["mia_gap_match_abs"],
                    row["mia_auc_match_abs"],
                    row["mia_rank_match_abs"],
                    row["predictive_kl"],
                    row["test_f1_gap_abs"],
                    row["retained_val_loss"],
                ),
            )
        )
    return selected


def build_rule_report(name, selected_rows, extra=None):
    report = {
        "name": name,
        "aggregate": aggregate_selected_rows(selected_rows),
        "selected_configs": [],
    }
    for row in selected_rows:
        item = {
            "client": int(row["client"]),
            "eta": float(row["eta"]),
            "damping": float(row["damping"]),
            "output_layer_scale": float(row.get("output_layer_scale", 1.0)),
            "hidden_layer_scale": float(row.get("hidden_layer_scale", 1.0)),
            "summary_window": int(row.get("summary_window", 0)),
            "step_l2": float(row["step_l2"]),
            "retained_val_loss": float(row["retained_val_loss"]),
            "predictive_kl": float(row["predictive_kl"]),
            "test_acc_gap_abs": float(row["test_acc_gap_abs"]),
            "test_auc_gap_abs": float(row["test_auc_gap_abs"]),
            "test_f1_gap_abs": float(row["test_f1_gap_abs"]),
        }
        if "mia_candidate_auc" in row:
            item["mia_candidate_auc"] = float(row["mia_candidate_auc"])
            item["mia_candidate_gap"] = float(row["mia_candidate_gap"])
            item["mia_candidate_rank"] = float(row["mia_candidate_rank"])
            item["mia_auc_match_abs"] = float(row["mia_auc_match_abs"])
            item["mia_gap_match_abs"] = float(row["mia_gap_match_abs"])
            item["mia_rank_match_abs"] = float(row["mia_rank_match_abs"])
        report["selected_configs"].append(item)
    if extra is not None:
        report["selector_config"] = extra
    return report


def meets_global_acceptance(report):
    agg = report["aggregate"]
    if "mia_auc_match_abs" not in agg:
        return False
    return (
        agg["mia_auc_match_abs"] <= 0.10
        and agg["mia_gap_match_abs"] <= 0.18
        and agg["mia_rank_match_abs"] <= 1.0
        and agg["predictive_kl"] <= 0.045
        and agg["test_acc_gap_abs"] <= 0.017
        and agg["test_auc_gap_abs"] <= 0.014
        and agg["test_f1_gap_abs"] <= 0.045
    )


def compute_outlier_improvement(center_rows, selected_rows):
    center_by_client = {int(row["client"]): row for row in center_rows}
    selected_by_client = {int(row["client"]): row for row in selected_rows}
    improved_clients = []
    for client in OUTLIER_CLIENTS:
        before = center_by_client[client]
        after = selected_by_client[client]
        if (
            after["mia_auc_match_abs"] < before["mia_auc_match_abs"]
            and after["predictive_kl"] <= before["predictive_kl"] + 0.01
        ):
            improved_clients.append(client)
    return {
        "improved_clients": improved_clients,
        "improvement_count": len(improved_clients),
    }


def acceptance_deficit(report):
    agg = report["aggregate"]
    targets = {
        "mia_auc_match_abs": 0.10,
        "mia_gap_match_abs": 0.18,
        "mia_rank_match_abs": 1.0,
        "predictive_kl": 0.045,
        "test_acc_gap_abs": 0.017,
        "test_auc_gap_abs": 0.014,
        "test_f1_gap_abs": 0.045,
    }
    misses = 0
    score = 0.0
    for key, target in targets.items():
        value = float(agg.get(key, float("inf")))
        if value > target:
            misses += 1
            score += (value - target) / target
    return misses, score


def choose_better_match_report(current_report, candidate_report):
    current_ok = meets_global_acceptance(current_report)
    candidate_ok = meets_global_acceptance(candidate_report)
    if candidate_ok and not current_ok:
        return candidate_report
    if current_ok and not candidate_ok:
        return current_report

    current_misses, current_deficit = acceptance_deficit(current_report)
    candidate_misses, candidate_deficit = acceptance_deficit(candidate_report)
    if candidate_misses < current_misses:
        return candidate_report
    if candidate_misses > current_misses:
        return current_report
    if candidate_deficit < current_deficit - 1e-12:
        return candidate_report
    if current_deficit < candidate_deficit - 1e-12:
        return current_report
    if candidate_report["aggregate"]["mia_gap_match_abs"] < current_report["aggregate"]["mia_gap_match_abs"] - 1e-12:
        return candidate_report
    return current_report


def expand_recent_summary_windows(rows_by_client, center_rows, evaluate_candidate_row):
    expanded = {client: list(rows) for client, rows in rows_by_client.items()}
    center_by_client = {int(row["client"]): row for row in center_rows}
    for client in OUTLIER_CLIENTS:
        existing = {candidate_key(row) for row in expanded[client]}
        center = center_by_client[client]
        progress(f"testing recent summaries for client {client}")
        for window in RECENT_SUMMARY_WINDOWS:
            key = candidate_key(
                {
                    "client": client,
                    "eta": float(center["eta"]),
                    "damping": float(center["damping"]),
                    "output_layer_scale": float(center.get("output_layer_scale", 1.0)),
                    "hidden_layer_scale": float(center.get("hidden_layer_scale", 1.0)),
                    "summary_window": int(window),
                }
            )
            if key in existing:
                continue
            row = evaluate_candidate_row(
                client,
                float(center["eta"]),
                float(center["damping"]),
                output_layer_scale=float(center.get("output_layer_scale", 1.0)),
                hidden_layer_scale=float(center.get("hidden_layer_scale", 1.0)),
                summary_window=int(window),
            )
            expanded[client].append(row)
            existing.add(key)
    return expanded


def expand_local_neighborhood(rows_by_client, center_config, evaluate_candidate_row, hidden_scales):
    expanded = {client: list(rows) for client, rows in rows_by_client.items()}
    for client in OUTLIER_CLIENTS:
        existing = {candidate_key(row) for row in expanded[client]}
        center = center_config[client]
        progress(f"expanding local search for client {client}")
        for eta_multiplier in ETA_MULTIPLIERS:
            eta = float(center["eta"]) * eta_multiplier
            for damping_offset in DAMPING_OFFSETS:
                damping = min(0.25, max(0.05, float(center["damping"]) + damping_offset))
                for output_scale in LOCAL_OUTPUT_SCALES:
                    for hidden_scale in hidden_scales:
                        key = candidate_key(
                            {
                                "client": client,
                                "eta": eta,
                                "damping": damping,
                                "output_layer_scale": output_scale,
                                "hidden_layer_scale": hidden_scale,
                                "summary_window": int(center.get("summary_window", 0)),
                            }
                        )
                        if key in existing:
                            continue
                        row = evaluate_candidate_row(
                            client,
                            eta,
                            damping,
                            output_layer_scale=output_scale,
                            hidden_layer_scale=hidden_scale,
                            summary_window=int(center.get("summary_window", 0)),
                        )
                        expanded[client].append(row)
                        existing.add(key)
    return expanded


def main():

    """
    Required arguments:
    * base-dir  : should contain 'partitions.pkl', the original model's 
    parameters, hessian and summaries and the retrained models' parameters,
    hessians and summaries
    * data-dir  : should contain 'train_data.pkl' and 'test_data.pkl'
    * output-dir: the results of the experiments will be written to this folder

    What it does:
    * Run the unlearning procedure with different client-level unlearning 
    hyperparameter configurations
    * Apply seven different rules for selecting a configuration 
    * Pick a center configuration
    * Test MIA performance around this center configuration
    * Score the models
    """

    # Convert input strings to numeric lists
    args = parse_args()
    eta_grid = parse_float_grid(args.eta_grid)
    damping_grid = parse_float_grid(args.damping_grid)
    floor_ratios = parse_float_grid(args.floor_ratios)
    scale_factors = parse_float_grid(args.scale_factors)
    loss_band_eps = parse_float_grid(args.loss_band_eps)
    guarded_auc_tols = parse_float_grid(args.guarded_auc_tols)
    guarded_ece_tols = parse_float_grid(args.guarded_ece_tols)
    score_loss_weights = parse_float_grid(args.score_loss_weights)
    score_auc_weights = parse_float_grid(args.score_auc_weights)
    score_ece_weights = parse_float_grid(args.score_ece_weights)
    score_step_weights = parse_float_grid(args.score_step_weights)
    output_scales = parse_float_grid(args.output_scales)

    # Set up output directory and cache file, load MIA report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "selector_cache.json"
    client_mia_report = load_client_mia_report(args.client_mia_report)

    # The first config from the file is treated as the baseline config
    with open(Path(args.base_dir) / "unlearning_config.json", "r", encoding="utf-8") as handle:
        base_config = json.load(handle) 
    first_config = next(iter(base_config.values()))
    baseline_eta = float(first_config.get("eta", first_config.get("selected_eta")))
    baseline_damping = float(first_config["damping"])
    baseline_output_scale = float(first_config.get("output_layer_scale", 1.0))


    # If the same experiment has been done already, use that cache
    cache = None
    if cache_path.exists() and not args.force:
        with open(cache_path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        cached_config = cached.get("config", {})
        if (
            cached_config.get("eta_grid") == eta_grid
            and cached_config.get("damping_grid") == damping_grid
            and cached_config.get("output_scales", [1.0]) == output_scales
            and int(cached_config.get("batch_size", args.batch_size)) == args.batch_size
            and int(cached_config.get("seed", args.seed)) == args.seed
        ):
            progress(f"loading cached selector grid from {cache_path}")
            cache = cached
        else:
            progress("cached selector grid does not match current search config, rebuilding")

    # Build a andidate cache by running the unlearning procedure for the
    # parameter combinations prescribed by the grid and store per-client
    # metrics on performance and difference with retrained model
    if cache is None:
        cache = build_candidate_cache(
            base_dir=args.base_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            eta_grid=eta_grid,
            damping_grid=damping_grid,
            output_scales=output_scales,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    # Ensure all rows have the same data type for each feature
    rows_by_client = {
        int(client): [normalize_candidate_row(row) for row in rows]
        for client, rows in cache["candidate_rows"].items()
    }

    # Construct two helper functions:
    # - eval... : builds unlearned model for specified config, evaluates 
    # it and computes predictive metrics
    # - attach... : if row has no MIA features, build the unlearned model,
    # run the attack for each seed, average to get auc, gap and rank
    evaluate_candidate_row, attach_mia_metrics = build_candidate_evaluator(
        base_dir=args.base_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        client_mia_report=client_mia_report,
    )

    # For all the candidates with client_mia_report, attach the metrics
    if client_mia_report is not None:
        progress("attaching client mia metrics to cached candidates")
        total_rows = sum(len(rows) for rows in rows_by_client.values())
        attached = 0
        for client, rows in rows_by_client.items():
            for index, row in enumerate(rows):
                if "mia_candidate_auc" in row:
                    attached += 1
                    continue
                rows_by_client[client][index] = attach_mia_metrics(row)
                attached += 1
                if attached % 50 == 0 or attached == total_rows:
                    progress(f"attached client mia metrics to {attached}/{total_rows} cached candidates")

    # Pick the unlearning config that was previously selected as baseline
    baseline_rows = select_baseline(
        rows_by_client,
        eta=baseline_eta,
        damping=baseline_damping,
        output_layer_scale=baseline_output_scale,
    )


    # The following part executes different rules, each selecting a 
    # configuration for each of the clients


    # Pick the row that minimizes test acc gap, test auc gap, test f1 gap
    # (Importance is in that order)
    oracle_rows = select_oracle(rows_by_client)

    # Pick the row that minimizes the retained validation loss, maximizes
    # retained validation accuracy and minimizes retained vall ece (importance
    # in that order)
    retained_rows = select_retained_loss(rows_by_client)

    reports = {
        "baseline": build_rule_report(
            "baseline",
            baseline_rows,
            {
                "eta": baseline_eta,
                "damping": baseline_damping,
                "output_layer_scale": baseline_output_scale,
            },
        ),
        "oracle_kl": build_rule_report("oracle_kl", oracle_rows),
        "retained_loss": build_rule_report("retained_loss", retained_rows),
    }

    # Testing different rules

    # 1) Retained-loss step floors
    # The L2 distance from the og has to be at least proportional to that
    # of the baseline. Various ratios are tested, as prescribed by floor_ratios.
    # Among the rmaining rows, choose the oine with best retained
    # loss + tiebreakers.
    progress("testing retained-loss selector with step floors")
    floor_reports = []
    for ratio in floor_ratios:
        floor_reports.append(
            build_rule_report(
                f"retained_loss_floor_{ratio}",
                select_retained_loss_with_floor(rows_by_client, baseline_rows, ratio),
                {"floor_ratio": ratio},
            )
        )
    floor_reports.sort(key=lambda item: item["aggregate"]["retained_val_loss"])
    reports["retained_loss_step_floor_candidates"] = floor_reports
    reports["retained_loss_step_floor_best"] = floor_reports[0]
    
    # 2) Retained-loss safe bands
    # Determine the smallest loss, keep only the rows within a band around
    # that. Out of those, choose the one with the largest step size. Various
    # band sizes are tested, as given in loss_band_eps.
    progress("testing retained-loss safe bands")
    band_reports = []
    for eps in loss_band_eps:
        band_reports.append(
            build_rule_report(
                f"retained_loss_safe_band_{eps}",
                select_loss_band_largest_step(rows_by_client, eps),
                {"loss_eps": eps},
            )
        )
    band_reports.sort(key=lambda item: item["aggregate"]["predictive_kl"])
    reports["retained_loss_safe_band_candidates"] = band_reports
    reports["retained_loss_safe_band_best"] = band_reports[0]

    # 3) Guarded retained-loss bands
    # Determine the smallest loss, highest AUC and lowest ECE, then keep
    # only rows within bands around those.  If this does not exist, relax 
    # the constraints and keep only loss band. If still empty, keep all 
    # rows. From those acceptable candidates, keep again the one with the largest step size.
    
    progress("testing guarded retained-loss bands")
    guarded_reports = []
    for loss_eps in loss_band_eps:
        for auc_tol in guarded_auc_tols:
            for ece_tol in guarded_ece_tols:
                guarded_reports.append(
                    build_rule_report(
                        "guarded_loss_band",
                        select_guarded_loss_band(rows_by_client, loss_eps, auc_tol, ece_tol),
                        {
                            "loss_eps": loss_eps,
                            "auc_tol": auc_tol,
                            "ece_tol": ece_tol,
                        },
                    )
                )
    guarded_reports.sort(
        key=lambda item: (
            item["aggregate"]["predictive_kl"],
            item["aggregate"]["test_f1_gap_abs"],
            item["aggregate"]["test_acc_gap_abs"],
            item["aggregate"]["test_auc_gap_abs"],
        )
    )
    reports["guarded_loss_band_candidates"] = guarded_reports
    reports["guarded_loss_band_best"] = guarded_reports[0]


    # 4) Step-norm selectors
    # Pick the rows with step size (L2 norm) closer to a target step size.
    
    # Compute a typical baseline L2 step 
    baseline_step_l2 = np.median([row["step_l2"] for row in baseline_rows])
    # How much the candidate is allowed to stray from this baseline
    step_targets = [baseline_step_l2 * factor for factor in scale_factors]
    progress("testing step-norm selectors")
    step_reports = []
    # only consider rows what that damping factor
    for damping in damping_grid:
        for target in step_targets:
            step_reports.append(
                build_rule_report(
                    "step_norm",
                    select_step_norm_rule(rows_by_client, damping, target),
                    {"damping": damping, "target_step_l2": target},
                )
            )
    step_reports.sort(key=lambda item: item["aggregate"]["retained_val_loss"])
    reports["step_norm_candidates"] = step_reports
    reports["step_norm_best"] = step_reports[0]

    # 5) Summary-norm selector
    # Similarly, pick the one with eta closest to the ideal eta, which is
    # scale / L2 of the summary
    summary_l2_median = np.median([rows[0]["summary_l2"] for rows in rows_by_client.values()])
    summary_scales = [summary_l2_median * baseline_eta * factor for factor in scale_factors]
    progress("testing summary-norm selectors")
    summary_reports = []
    for damping in damping_grid:
        for scale in summary_scales:
            summary_reports.append(
                build_rule_report(
                    "summary_norm",
                    select_summary_norm_rule(rows_by_client, damping, scale),
                    {"damping": damping, "scale": scale},
                )
            )
    summary_reports.sort(key=lambda item: item["aggregate"]["retained_val_loss"])
    reports["summary_norm_candidates"] = summary_reports
    reports["summary_norm_best"] = summary_reports[0]

    # 6 ) Balanced retained-score selctor
    #  Normalize the metrics so that higher is always better. Then for 
    # different combinations of weights, calculate a score for each 
    # client row
    progress("testing balanced retained-score selectors")
    score_reports = []
    for loss_weight in score_loss_weights:
        for auc_weight in score_auc_weights:
            for ece_weight in score_ece_weights:
                for step_weight in score_step_weights:
                    score_reports.append(
                        build_rule_report(
                            "balanced_retained_score",
                            select_balanced_score_rule(
                                rows_by_client,
                                loss_weight,
                                auc_weight,
                                ece_weight,
                                step_weight,
                            ),
                            {
                                "loss_weight": loss_weight,
                                "auc_weight": auc_weight,
                                "ece_weight": ece_weight,
                                "step_weight": step_weight,
                            },
                        )
                    )
    score_reports.sort(
        key=lambda item: (
            item["aggregate"]["predictive_kl"],
            item["aggregate"]["test_f1_gap_abs"],
            item["aggregate"]["test_acc_gap_abs"],
            item["aggregate"]["test_auc_gap_abs"],
        )
    )
    reports["balanced_score_candidates"] = score_reports
    reports["balanced_score_best"] = score_reports[0]

    # Collect the results of all the different rules
    practical_candidates = [
        reports["retained_loss"],
        reports["retained_loss_step_floor_best"],
        reports["retained_loss_safe_band_best"],
        reports["guarded_loss_band_best"],
        reports["step_norm_best"],
        reports["summary_norm_best"],
        reports["balanced_score_best"],
    ]

    # Choose the best out of those seven rules, based on four metrics
    practical_candidates.sort(
        key=lambda item: (
            item["aggregate"]["predictive_kl"],
            item["aggregate"]["test_acc_gap_abs"],
            item["aggregate"]["test_auc_gap_abs"],
            item["aggregate"]["test_f1_gap_abs"],
        )
    )
    reports["best_practical_by_kl"] = practical_candidates[0]


    if client_mia_report is not None:

        # Filters out rows that do not pass retrain match guardrails (hardcoded)
        # Build normalized MIA scores, utility metrics
        # Combine into weighted sum, pick the lowest score
        progress("testing retrain-match selector on cache")
        cache_match_rows = select_retrain_match_rule(
            rows_by_client,
            mia_auc_weight=args.mia_auc_weight,
            mia_gap_weight=args.mia_gap_weight,
            mia_rank_weight=args.mia_rank_weight,
        )
        reports["retrain_match_score_cache"] = build_rule_report(
            "retrain_match_score_cache",
            cache_match_rows,
            {
                "mia_auc_weight": args.mia_auc_weight,
                "mia_gap_weight": args.mia_gap_weight,
                "mia_rank_weight": args.mia_rank_weight,
            },
        )

        selected_config_path = client_mia_report["config"].get("selected_config_path")
        center_config = None

        # If a center was specified, use it
        if selected_config_path:
            center_config = load_external_config(selected_config_path)
            center_rows = materialize_config_rows(rows_by_client, center_config, evaluate_candidate_row)
            reports["balanced_score_plus_output_scale"] = build_rule_report(
                "balanced_score_plus_output_scale",
                center_rows,
                {"selected_config_path": selected_config_path},
            )
        
        # Else, take the one with the best balanced score as center
        else:
            center_rows = reports["balanced_score_best"]["selected_configs"]
            center_config = {
                int(row["client"]): {
                    "eta": float(row["eta"]),
                    "damping": float(row["damping"]),
                    "output_layer_scale": float(row.get("output_layer_scale", 1.0)),
                    "hidden_layer_scale": float(row.get("hidden_layer_scale", 1.0)),
                    "summary_window": int(row.get("summary_window", 0)),
                }
                for row in center_rows
            }

        # Try small perturbations around the center configuration for each
        # client. Evaluate these if not done yet. 
        progress("running local retrain-match neighborhood search")
        local_rows_by_client = expand_local_neighborhood(
            rows_by_client,
            center_config,
            evaluate_candidate_row,
            hidden_scales=[1.0],
        )

        # Then select a config using the select_retrain_match_rule.
        local_match_rows = select_retrain_match_rule(
            local_rows_by_client,
            mia_auc_weight=args.mia_auc_weight,
            mia_gap_weight=args.mia_gap_weight,
            mia_rank_weight=args.mia_rank_weight,
        )

        # Build the report
        local_report = build_rule_report(
            "retrain_match_score_local",
            local_match_rows,
            {
                "mia_auc_weight": args.mia_auc_weight,
                "mia_gap_weight": args.mia_gap_weight,
                "mia_rank_weight": args.mia_rank_weight,
                "outlier_clients": OUTLIER_CLIENTS,
                "eta_multipliers": ETA_MULTIPLIERS,
                "damping_offsets": DAMPING_OFFSETS,
                "output_scales": LOCAL_OUTPUT_SCALES,
            },
        )
        
        # Compute outlier improvement. 
        local_report["outlier_improvement"] = compute_outlier_improvement(center_rows, local_match_rows)
        reports["retrain_match_score_local"] = local_report
        best_retrain_match_report = local_report

        # If this did not meet acceptance threshold, try again, now expanding
        # hidden layer scales as well
        if not (
            meets_global_acceptance(local_report)
            and local_report["outlier_improvement"]["improvement_count"] >= 4
        ):
            progress("local pass missed targets, adding hidden-layer scale search")
            hidden_rows_by_client = expand_local_neighborhood(
                rows_by_client,
                center_config,
                evaluate_candidate_row,
                hidden_scales=HIDDEN_LAYER_SCALES,
            )
            hidden_match_rows = select_retrain_match_rule(
                hidden_rows_by_client,
                mia_auc_weight=args.mia_auc_weight,
                mia_gap_weight=args.mia_gap_weight,
                mia_rank_weight=args.mia_rank_weight,
            )
            hidden_report = build_rule_report(
                "retrain_match_score_hidden",
                hidden_match_rows,
                {
                    "mia_auc_weight": args.mia_auc_weight,
                    "mia_gap_weight": args.mia_gap_weight,
                    "mia_rank_weight": args.mia_rank_weight,
                    "outlier_clients": OUTLIER_CLIENTS,
                    "eta_multipliers": ETA_MULTIPLIERS,
                    "damping_offsets": DAMPING_OFFSETS,
                    "output_scales": LOCAL_OUTPUT_SCALES,
                    "hidden_layer_scales": HIDDEN_LAYER_SCALES,
                },
            )
            hidden_report["outlier_improvement"] = compute_outlier_improvement(center_rows, hidden_match_rows)
            reports["retrain_match_score_hidden"] = hidden_report
            best_retrain_match_report = hidden_report

        # If this still does not meet the acceptance threshold, try different 
        # summary window values
        if not meets_global_acceptance(best_retrain_match_report):
            progress("hidden pass still missed targets, testing recent-round summaries")
            recent_rows_by_client = expand_recent_summary_windows(
                rows_by_client,
                best_retrain_match_report["selected_configs"],
                evaluate_candidate_row,
            )
            recent_match_rows = select_retrain_match_rule(
                recent_rows_by_client,
                mia_auc_weight=args.mia_auc_weight,
                mia_gap_weight=args.mia_gap_weight,
                mia_rank_weight=args.mia_rank_weight,
            )
            recent_report = build_rule_report(
                "retrain_match_score_recent",
                recent_match_rows,
                {
                    "mia_auc_weight": args.mia_auc_weight,
                    "mia_gap_weight": args.mia_gap_weight,
                    "mia_rank_weight": args.mia_rank_weight,
                    "outlier_clients": OUTLIER_CLIENTS,
                    "recent_summary_windows": RECENT_SUMMARY_WINDOWS,
                },
            )
            recent_report["outlier_improvement"] = compute_outlier_improvement(
                best_retrain_match_report["selected_configs"],
                recent_match_rows,
            )
            reports["retrain_match_score_recent"] = recent_report
            best_retrain_match_report = choose_better_match_report(best_retrain_match_report, recent_report)

        # Pick the most expanded pool available
        constrained_pool = rows_by_client
        if "retrain_match_score_recent" in reports:
            constrained_pool = recent_rows_by_client
        elif "retrain_match_score_hidden" in reports:
            constrained_pool = hidden_rows_by_client
        elif "retrain_match_score_local" in reports:
            constrained_pool = local_rows_by_client

        # Select, per client, configurations that satisfy hard constraints
        constrained_rows = select_constrained_retrain_match_rule(constrained_pool)
        reports["constrained_retrain_match"] = build_rule_report(
            "constrained_retrain_match",
            constrained_rows,
            {
                "predictive_kl_max": 0.05,
                "test_acc_gap_abs_max": 0.02,
                "test_auc_gap_abs_max": 0.02,
                "test_f1_gap_abs_max": 0.05,
                "mia_candidate_auc_min": 0.50,
            },
        )

        reports["best_retrain_match"] = best_retrain_match_report

    progress("writing selector study reports")
    save_json(reports, output_dir / "selector_reports.json")
    export_config_artifact(
        reports["retained_loss_safe_band_best"],
        output_dir / "best_safe_band_configs.json",
    )
    export_config_artifact(
        reports["best_practical_by_kl"],
        output_dir / "best_practical_configs.json",
    )
    if "best_retrain_match" in reports:
        export_config_artifact(
            reports["best_retrain_match"],
            output_dir / "best_retrain_match_configs.json",
        )

    progress("selector summary")
    summary_keys = [
        "baseline",
        "oracle_kl",
        "retained_loss",
        "retained_loss_step_floor_best",
        "retained_loss_safe_band_best",
        "guarded_loss_band_best",
        "step_norm_best",
        "summary_norm_best",
        "balanced_score_best",
        "best_practical_by_kl",
    ]
    if "balanced_score_plus_output_scale" in reports:
        summary_keys.append("balanced_score_plus_output_scale")
    if "retrain_match_score_cache" in reports:
        summary_keys.append("retrain_match_score_cache")
    if "retrain_match_score_local" in reports:
        summary_keys.append("retrain_match_score_local")
    if "retrain_match_score_hidden" in reports:
        summary_keys.append("retrain_match_score_hidden")
    if "retrain_match_score_recent" in reports:
        summary_keys.append("retrain_match_score_recent")
    if "constrained_retrain_match" in reports:
        summary_keys.append("constrained_retrain_match")
    if "best_retrain_match" in reports:
        summary_keys.append("best_retrain_match")

    for key in summary_keys:
        aggregate = reports[key]["aggregate"]
        line = (
            f"{key}: kl={aggregate['predictive_kl']:.4f} "
            f"acc_gap={aggregate['test_acc_gap_abs']:.4f} "
            f"auc_gap={aggregate['test_auc_gap_abs']:.4f} "
            f"f1_gap={aggregate['test_f1_gap_abs']:.4f}"
        )
        if "mia_auc_match_abs" in aggregate:
            line += (
                f" mia_auc={aggregate['mia_auc_match_abs']:.4f} "
                f"mia_gap={aggregate['mia_gap_match_abs']:.4f} "
                f"mia_rank={aggregate['mia_rank_match_abs']:.4f}"
            )
        progress(line)


if __name__ == "__main__":
    main()
