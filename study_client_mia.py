import argparse
import json
import pickle
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
from data_utils import load_pkl_as_dataset
from training import Net
from unlearning import unlearn
from study_unlearning_selectors import build_recent_summary_bank, ensure_original_round_deltas


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-path", default="train_data.pkl")
    parser.add_argument("--metadata-path", default="feature_metadata.json")
    parser.add_argument("--selected-config-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-bags", type=int, default=12)
    parser.add_argument("--cross-negative-bags", type=int, default=4)
    parser.add_argument("--eval-bags", type=int, default=10)
    parser.add_argument("--bag-size", type=int, default=256)
    parser.add_argument("--attack-c", type=float, default=None)
    parser.add_argument("--attack-c-grid", default="0.1,0.2,0.5,1.0")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--report-key-prefix", default=None)
    return parser.parse_args()


def progress(message):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_int_grid(text):
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def parse_float_grid(text):
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def resolve_path(base_dir, path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def load_selected_config(path):
    if path is None:
        return None

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


def build_model_from_params(params, input_size):
    model = Net(input_size=input_size)
    model.load_state_dict(params)
    return model


def make_client_eval_loaders(train_dataset, client_train_idx, batch_size):
    loaders = []
    for indices in client_train_idx:
        subset = Subset(train_dataset, [int(idx) for idx in indices])
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=False))
    return loaders


def metric_mean_std(rows, metric):
    values = [float(row[metric]) for row in rows]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def aggregate_rows(rows):
    aggregate = {}
    for metric in [
        "auc",
        "client_auc",
        "deleted_score",
        "retained_mean_score",
        "score_gap",
        "score_z",
        "quantile",
        "quantile_gap",
        "client_deleted_score",
        "client_retained_mean_score",
        "client_score_gap",
        "rank",
    ]:
        values = [float(row[metric]) for row in rows]
        aggregate[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    aggregate["top1_count"] = int(sum(1 for row in rows if int(row["rank"]) == 1))
    aggregate["top3_count"] = int(sum(1 for row in rows if int(row["rank"]) <= 3))
    aggregate["auc_gt_half_count"] = int(sum(1 for row in rows if float(row["auc"]) > 0.5))
    aggregate["gap_positive_count"] = int(sum(1 for row in rows if float(row["score_gap"]) > 0.0))
    return aggregate


def average_metric_rows(rows):
    averaged = {}
    for metric in [
        "auc",
        "client_auc",
        "deleted_score",
        "retained_mean_score",
        "score_gap",
        "score_z",
        "quantile",
        "quantile_gap",
        "client_deleted_score",
        "client_retained_mean_score",
        "client_score_gap",
        "rank",
    ]:
        values = [float(row[metric]) for row in rows]
        averaged[metric] = float(np.mean(values))
        averaged[f"{metric}_std"] = float(np.std(values))
    return averaged


def choose_attack_c_report(attack_c_reports):
    def metric_mean(report, label, metric_name):
        metrics = report["aggregate"][label]
        if metric_name == "bag_auc":
            metric_name = "auc"
        return float(metrics[metric_name]["mean"])

    kept = [
        report
        for report in attack_c_reports
        if 0.48 <= metric_mean(report, "original", "client_auc") <= 0.52
    ]
    if not kept:
        kept = sorted(
            attack_c_reports,
            key=lambda report: (
                abs(metric_mean(report, "original", "client_auc") - 0.5),
                -metric_mean(report, "retrain", "client_auc"),
                report["attack_c"],
            ),
        )
        return kept[0]

    kept = sorted(
        kept,
        key=lambda report: (
            -metric_mean(report, "retrain", "client_auc"),
            report["attack_c"],
        ),
    )
    return kept[0]


def build_seed_attack_suite(
    original_rows,
    retrain_rows,
    seed,
    attack_c_grid,
    train_bags,
    cross_negative_bags,
    eval_bags,
    bag_size,
):
    attack_c_reports = []

    for attack_c in attack_c_grid:
        attacks = {}
        train_rows = []
        original_eval_rows = []
        retrain_eval_rows = []

        for holdout_client in range(len(original_rows)):
            attack_model, train_metrics = train_client_deletion_attack(
                original_rows,
                retrain_rows,
                holdout_client,
                train_bags=train_bags,
                cross_negative_bags=cross_negative_bags,
                bag_size=bag_size,
                attack_c=attack_c,
                seed=seed,
            )
            attacks[holdout_client] = attack_model
            train_rows.append(train_metrics)
            original_eval_rows.append(
                score_client_deletion_attack(
                    attack_model,
                    original_rows,
                    original_rows,
                    holdout_client,
                    eval_bags=eval_bags,
                    bag_size=bag_size,
                    seed=seed,
                )
            )
            retrain_eval_rows.append(
                score_client_deletion_attack(
                    attack_model,
                    original_rows,
                    retrain_rows[holdout_client],
                    holdout_client,
                    eval_bags=eval_bags,
                    bag_size=bag_size,
                    seed=seed,
                )
            )

        attack_c_reports.append(
            {
                "seed": int(seed),
                "attack_c": float(attack_c),
                "attacks": attacks,
                "train_rows": train_rows,
                "aggregate": {
                    "train_auc": metric_mean_std(train_rows, "train_auc"),
                    "train_acc" : metric_mean_std(train_rows, "acc"), # testing this out
                    "original": aggregate_rows(original_eval_rows),
                    "retrain": aggregate_rows(retrain_eval_rows),
                },
            }
        )

    return choose_attack_c_report(attack_c_reports), attack_c_reports


def score_label_with_seed_suite(seed_runs, original_rows, candidate_rows_by_client, eval_bags, bag_size):
    rows_by_client = {client: [] for client in range(len(original_rows))}

    for seed_run in seed_runs:
        seed = int(seed_run["seed"])
        for holdout_client in range(len(original_rows)):
            row = score_client_deletion_attack(
                seed_run["attacks"][holdout_client],
                original_rows,
                candidate_rows_by_client[holdout_client],
                holdout_client,
                eval_bags=eval_bags,
                bag_size=bag_size,
                seed=seed,
            )
            rows_by_client[holdout_client].append(row)

    averaged_rows = []
    per_client = {}
    for holdout_client in range(len(original_rows)):
        averaged = average_metric_rows(rows_by_client[holdout_client])
        per_client[str(holdout_client)] = averaged
        averaged_rows.append(averaged)

    return per_client, aggregate_rows(averaged_rows)


def score_labels_with_seed_suite(seed_runs, original_rows, labels_to_score, eval_bags, bag_size):
    report_per_client = {str(client): {} for client in range(len(original_rows))}
    report_aggregate = {}

    train_auc_rows = [
        {"train_auc": float(seed_run["aggregate"]["train_auc"]["mean"])}
        for seed_run in seed_runs
    ]
    report_aggregate["train_auc"] = metric_mean_std(train_auc_rows, "train_auc")

    for label, candidate_rows_by_client in labels_to_score.items():
        per_client, aggregate = score_label_with_seed_suite(
            seed_runs,
            original_rows,
            candidate_rows_by_client,
            eval_bags=eval_bags,
            bag_size=bag_size,
        )
        report_aggregate[label] = aggregate
        for client in range(len(original_rows)):
            report_per_client[str(client)][label] = per_client[str(client)]

    return report_per_client, report_aggregate


def merge_report_into_output(output_path, report, report_key_prefix):
    if report_key_prefix is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return

    payload = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if isinstance(existing, dict) and "aggregate" in existing:
            payload["default"] = existing
        elif isinstance(existing, dict):
            payload.update(existing)

    payload[report_key_prefix] = report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    """
    Required argumetns:
    * experiment-dir : Should contain partitions, og/retr/unl parameters,
    hessian and summaries
    * data-dir       : Should contain the training and testing data

    """
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    data_dir = Path(args.data_dir)
    output_path = Path(args.output_path) if args.output_path else experiment_dir / "client_deletion_mia_report.json"
    selected_config = load_selected_config(args.selected_config_path)
    seeds = parse_int_grid(args.seeds)
    attack_c_grid = [float(args.attack_c)] if args.attack_c is not None else parse_float_grid(args.attack_c_grid)

    progress("loading data and partitions")
    train_path = resolve_path(data_dir, args.train_path)
    metadata_path = resolve_path(data_dir, args.metadata_path)
    train_dataset, metadata = load_pkl_as_dataset(train_path, metadata_path)
    input_size = metadata["input_size"]
    partitions = pickle.load(open(experiment_dir / "partitions.pkl", "rb"))
    client_train_idx = partitions["client_train_idx"]
    client_loaders = make_client_eval_loaders(train_dataset, client_train_idx, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress(f"using device {device}")

    progress("loading original model rows")
    og_params = torch.load(experiment_dir / "og_params.pt", map_location="cpu", weights_only=True)
    original_model = build_model_from_params(og_params, input_size)
    original_rows = build_client_feature_rows(original_model, client_loaders, device)

    progress("loading retrain rows")
    retrain_rows = []
    for removed_client in range(len(client_loaders)):
        retrain_params = torch.load(
            experiment_dir / f"{removed_client}" / "retr_params.pt",
            map_location="cpu",
            weights_only=True,
        )
        retrain_model = build_model_from_params(retrain_params, input_size)
        retrain_rows.append(build_client_feature_rows(retrain_model, client_loaders, device))

    progress("loading saved unlearn rows")
    saved_unlearn_rows = []
    for removed_client in range(len(client_loaders)):
        saved_params = torch.load(
            experiment_dir / f"{removed_client}" / "unl_params.pt",
            map_location="cpu",
            weights_only=True,
        )
        saved_model = build_model_from_params(saved_params, input_size)
        saved_unlearn_rows.append(build_client_feature_rows(saved_model, client_loaders, device))

    progress("loading finetune rows")
    finetune_rows = []
    for removed_client in range(len(client_loaders)):
        finetune_params = torch.load(
            experiment_dir / f"{removed_client}" / "ft_params.pt",
            map_location="cpu",
            weights_only=True,
        )
        finetune_model = build_model_from_params(finetune_params, input_size)
        finetune_rows.append(build_client_feature_rows(finetune_model, client_loaders, device))

    selected_unlearn_rows = None
    if selected_config is not None:
        progress("building selected unlearn rows from config")
        og_hessian = torch.load(experiment_dir / "og_hessian", map_location="cpu", weights_only=False)
        og_summaries = torch.load(experiment_dir / "og_summaries", map_location="cpu", weights_only=False)
        summary_bank = {0: og_summaries}
        selected_unlearn_rows = []
        for removed_client in range(len(client_loaders)):
            config = selected_config[removed_client]
            summary_window = int(config.get("summary_window", 0))
            if summary_window not in summary_bank:
                delta_rounds = ensure_original_round_deltas(experiment_dir, data_dir, args.batch_size, 0)
                summary_bank.update(build_recent_summary_bank(delta_rounds))
            params = unlearn(
                og_params,
                og_hessian,
                summary_bank.get(summary_window, og_summaries),
                removed_client,
                len(client_loaders),
                config["eta"],
                config["damping"],
                step_sign=1.0,
                output_layer_scale=config["output_layer_scale"],
                hidden_layer_scale=config["hidden_layer_scale"],
            )
            model = build_model_from_params(params, input_size)
            selected_unlearn_rows.append(build_client_feature_rows(model, client_loaders, device))

    labels_to_score = {
        "original": [original_rows for _ in range(len(client_loaders))],
        "retrain": retrain_rows,
        "unlearn_saved": saved_unlearn_rows,
        "finetune": finetune_rows,
    }
    if selected_unlearn_rows is not None:
        labels_to_score["unlearn_selected"] = selected_unlearn_rows

    progress("building seed attack suites")
    seed_runs = []
    seed_search_reports = {}
    for seed in seeds:
        progress(f"seed {seed}: searching attack c")
        selected_seed_run, all_reports = build_seed_attack_suite(
            original_rows,
            retrain_rows,
            seed=seed,
            attack_c_grid=attack_c_grid,
            train_bags=args.train_bags,
            cross_negative_bags=args.cross_negative_bags,
            eval_bags=args.eval_bags,
            bag_size=args.bag_size,
        )
        seed_runs.append(selected_seed_run)
        seed_search_reports[str(seed)] = {
            "selected_attack_c": float(selected_seed_run["attack_c"]),
            "candidates": [
                {
                    "attack_c": float(report["attack_c"]),
                    "train_auc": report["aggregate"]["train_auc"],
                    "original_bag_auc": report["aggregate"]["original"]["auc"],
                    "original_client_auc": report["aggregate"]["original"]["client_auc"],
                    "retrain_bag_auc": report["aggregate"]["retrain"]["auc"],
                    "retrain_client_auc": report["aggregate"]["retrain"]["client_auc"],
                }
                for report in all_reports
            ],
        }

    progress("scoring labels with selected seed suites")
    per_client_report, aggregate_report = score_labels_with_seed_suite(
        seed_runs,
        original_rows,
        labels_to_score,
        eval_bags=args.eval_bags,
        bag_size=args.bag_size,
    )

    for holdout_client in range(len(client_loaders)):
        per_client_report[str(holdout_client)]["train"] = {
            "train_auc": float(np.mean([seed_run["train_rows"][holdout_client]["train_auc"] for seed_run in seed_runs])),
            "train_auc_std": float(np.std([seed_run["train_rows"][holdout_client]["train_auc"] for seed_run in seed_runs])),
            "selected_attack_c_values": [float(seed_run["attack_c"]) for seed_run in seed_runs],
        }

    report = {
        "config": {
            "feature_mode": "delta_quantile",
            "attack_model": "scaled_logistic_regression",
            "attack_c": None if args.attack_c is None else float(args.attack_c),
            "attack_c_grid": attack_c_grid,
            "selected_attack_c_by_seed": [
                {
                    "seed": int(seed_run["seed"]),
                    "attack_c": float(seed_run["attack_c"]),
                }
                for seed_run in seed_runs
            ],
            "seeds": seeds,
            "train_bags": int(args.train_bags),
            "cross_negative_bags": int(args.cross_negative_bags),
            "eval_bags": int(args.eval_bags),
            "bag_size": int(args.bag_size),
            "experiment_dir": str(experiment_dir),
            "data_dir": str(data_dir),
            "selected_config_path": args.selected_config_path,
            "auc_metric": "client_auc",
            "gap_metric": "quantile_gap",
            "rank_metric": "client_rank",
            "selection_rule": "keep original client auc in [0.48, 0.52], then pick highest retrain client auc, tie to smaller attack_c",
            "note": "leave-one-client-out client deletion mia trained on retrain deltas from the other clients, using scaled logistic bags with client-level auc as the main forgetting score",
        },
        "attack_c_search": seed_search_reports,
        "per_seed": {
            f"seed_{seed_run['seed']}": {
                "selected_attack_c": float(seed_run["attack_c"]),
                "train_auc": seed_run["aggregate"]["train_auc"],
                "original": seed_run["aggregate"]["original"],
                "retrain": seed_run["aggregate"]["retrain"],
            }
            for seed_run in seed_runs
        },
        "per_client": per_client_report,
        "aggregate": aggregate_report,
    }

    merge_report_into_output(output_path, report, args.report_key_prefix)

    progress(f"wrote client mia report to {output_path}")
    progress(f"original client auc mean: {report['aggregate']['original']['client_auc']['mean']:.4f}")
    progress(f"retrain client auc mean: {report['aggregate']['retrain']['client_auc']['mean']:.4f}")
    progress(f"saved unlearn client auc mean: {report['aggregate']['unlearn_saved']['client_auc']['mean']:.4f}")
    if "unlearn_selected" in report["aggregate"]:
        progress(
            f"selected unlearn client auc mean: "
            f"{report['aggregate']['unlearn_selected']['client_auc']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()