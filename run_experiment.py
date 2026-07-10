import argparse
import pandas as pd
import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from data_utils import TabularDataset

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import (
    load_pkl_as_dataset,
    make_federated_loaders,
    make_global_loader_from_clients,
    save_default_partitions,
    save_processed_artifacts,
)
from federated import simulate_federated_learning
from preprocessing import build_audit, build_case_table, split_case_table, write_outputs
from reporting import build_unlearning_pair_summary
from training import Net, create_test_results_dict, create_val_results_dict, train_one_model
from unlearning import predictive_metrics, unlearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artificial", default=False)
    parser.add_argument("--xes-path", default=None)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--raw-train-path", default="raw_train_data.pkl")
    parser.add_argument("--raw-test-path", default="raw_test_data.pkl")
    parser.add_argument("--train-path", default="train_data.pkl")
    parser.add_argument("--test-path", default="test_data.pkl")
    parser.add_argument("--train-prop", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--unlearn-clients", default="all")
    parser.add_argument("--leniency", type=float, default=0.1)
    parser.add_argument("--min-client-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--fl-rounds", type=int, default=10)
    parser.add_argument("--fl-lr", type=float, default=0.1)
    parser.add_argument("--fl-optimizer", default="sgd")
    parser.add_argument("--fl-weight-decay", type=float, default=0.0)
    parser.add_argument("--fl-momentum", type=float, default=0.0)
    parser.add_argument("--fl-pos-weight", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--min-rounds", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=0.003)
    parser.add_argument("--unlearn-step-sign", type=float, default=1.0)
    parser.add_argument("--output-layer-scale", type=float, default=1.0)
    parser.add_argument("--per-client-config-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-traces", type=int, default=None)
    return parser.parse_args()


def resolve_path(base_dir, path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def save_pickle(obj, path):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def load_per_client_config(path):
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as handle:
        obj = json.load(handle)

    if isinstance(obj, dict) and "selected_configs" in obj:
        rows = obj["selected_configs"]
    elif isinstance(obj, list):
        rows = obj
    else:
        rows = obj.values()

    configs = {}
    for row in rows:
        client = int(row["client"])
        configs[client] = {
            "eta": float(row["eta"]),
            "damping": float(row["damping"]),
            "output_layer_scale": float(row.get("output_layer_scale", 1.0)),
        }
    return configs


def progress(message):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def parse_unlearn_clients(value, num_clients):
    if value == "all":
        return list(range(num_clients))

    clients = []
    for token in value.split(","):
        token = token.strip()
        if token:
            clients.append(int(token))
    return clients


def build_model_from_params(params, input_size):
    model = Net(input_size=input_size)
    model.load_state_dict(params)
    return model


def ensure_data_artifacts(args):
    data_dir = Path(args.data_dir)
    raw_train_path = resolve_path(data_dir, args.raw_train_path)
    raw_test_path = resolve_path(data_dir, args.raw_test_path)
    train_path = resolve_path(data_dir, args.train_path)
    test_path = resolve_path(data_dir, args.test_path)

    have_all = all(path.exists() for path in [raw_train_path, raw_test_path, train_path, test_path])
    if have_all:
        return raw_train_path, raw_test_path, train_path, test_path

    if args.xes_path is None:
        raise FileNotFoundError("missing data artifacts and no --xes-path was given")

    case_table = build_case_table(Path(args.xes_path), max_traces=args.max_traces)
    train_df, test_df = split_case_table(case_table, args.train_prop, args.seed)
    audit = build_audit(case_table, train_df, test_df)
    write_outputs(case_table, train_df, test_df, audit, data_dir)
    save_processed_artifacts(
        raw_train_path=raw_train_path,
        raw_test_path=raw_test_path,
        output_dir=data_dir,
    )

    return raw_train_path, raw_test_path, train_path, test_path


def aggregate_predictive_results(predictive_results):
    pair_keys = sorted(next(iter(predictive_results.values())).keys())
    metric_keys = sorted(next(iter(next(iter(predictive_results.values())).values())).keys())
    summary = {}

    for pair_key in pair_keys:
        summary[pair_key] = {}
        for metric_key in metric_keys:
            values = [float(predictive_results[client][pair_key][metric_key]) for client in predictive_results]
            summary[pair_key][metric_key] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }

    return summary


def aggregate_pair_summaries(summary_by_client):
    pair_keys = sorted(next(iter(summary_by_client.values())).keys())
    aggregate = {}

    for pair_key in pair_keys:
        pair_summary = {}
        first_entry = next(iter(summary_by_client.values()))[pair_key]

        for metric, value in first_entry["predictive"].items():
            if metric == "n_examples":
                continue
            series = [float(summary_by_client[client][pair_key]["predictive"][metric]) for client in summary_by_client]
            pair_summary[f"predictive_{metric}"] = {
                "mean": float(np.nanmean(series)),
                "std": float(np.nanstd(series)),
            }

        for scope, scope_values in first_entry["utility"]["val"].items():
            for metric in scope_values:
                series = [
                    float(summary_by_client[client][pair_key]["utility"]["val"][scope][metric]["gap"])
                    for client in summary_by_client
                ]
                pair_summary[f"{scope}_{metric}_gap"] = {
                    "mean": float(np.nanmean(series)),
                    "std": float(np.nanstd(series)),
                    "mean_abs": float(np.nanmean(np.abs(series))),
                }

        for metric in first_entry["utility"]["test"]:
            series = [
                float(summary_by_client[client][pair_key]["utility"]["test"][metric]["gap"])
                for client in summary_by_client
            ]
            pair_summary[f"test_{metric}_gap"] = {
                "mean": float(np.nanmean(series)),
                "std": float(np.nanstd(series)),
                "mean_abs": float(np.nanmean(np.abs(series))),
            }

        if first_entry["privacy"]:
            for metric in first_entry["privacy"]:
                series = [
                    float(summary_by_client[client][pair_key]["privacy"][metric]["gap"])
                    for client in summary_by_client
                ]
                pair_summary[f"privacy_{metric}_gap"] = {
                    "mean": float(np.nanmean(series)),
                    "std": float(np.nanstd(series)),
                    "mean_abs": float(np.nanmean(np.abs(series))),
                }

        aggregate[pair_key] = pair_summary

    return aggregate


def main():
    args = parse_args()
    set_seed(args.seed)
    progress("checking data artifacts")
    progress(
        f"unlearning config: eta={args.eta}, damping={args.damping}, "
        f"output_scale={args.output_layer_scale}"
    )
    per_client_config = load_per_client_config(args.per_client_config_path)
    if per_client_config is not None:
        progress(f"loaded per-client config from {args.per_client_config_path}")

    # raw_train_path, raw_test_path, train_path, test_path = ensure_data_artifacts(args)

    unlearn_clients = parse_unlearn_clients(args.unlearn_clients, args.num_clients)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"{args.min_client_size}-{args.alpha}-{args.damping}-{args.eta}-{args.fl_lr}-{args.num_clients}-{args.seed}-basic"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"writing outputs to {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress(f"using device {device}")

    path1 = f'./{args.data_dir}/train_data.pkl'
    path2 = f'./{args.data_dir}/feature_metadata.json'
    path3 = f'./{args.data_dir}/test_data.pkl'
    path4 = f'./{args.data_dir}/partitions.pkl'
    train_dataset, train_meta = load_pkl_as_dataset(path1, path2)
    test_dataset, _ = load_pkl_as_dataset(path3)
    # train_dataset, train_meta = load_pkl_as_dataset(train_path)
    # test_dataset, _ = load_pkl_as_dataset(test_path)
    input_size = train_meta["input_size"]
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    idx = pd.read_pickle(path4)
    client_train_idx = idx['client_train_idx']
    client_val_idx = idx['client_val_idx']
    # client_train_idx, client_val_idx, split_audit, kept_features = save_default_partitions(
    #     processed_train_path=train_path,
    #     raw_train_path=raw_train_path,
    #     output_dir=output_dir,
    #     num_clients=args.num_clients,
    #     seed=args.seed,
    #     leniency=args.leniency,
    #     min_client_size=args.min_client_size,
    #     train_prop=args.train_prop,
    # )
    progress(
        f"built {len(client_train_idx)} client partitions "
        # f"(min size {split_audit['min_client_size']}, max size {split_audit['max_client_size']})"
    )

    client_train_loaders, client_val_loaders = make_federated_loaders(
        train_dataset,
        client_train_idx,
        client_val_idx,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    global_train_loader, _ = make_global_loader_from_clients(
        train_dataset,
        client_train_idx,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
    )

    # Train the non-federated baseline
    non_fd_model = Net(input_size=input_size)
    progress("training non-federated baseline")
    train_one_model(
        non_fd_model,
        global_train_loader,
        device,
        epochs=args.rounds,
        lr=args.lr,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        pos_weight=args.pos_weight,
    )

    # Store the parameters, test and validation results
    # The test results are for the test loader as a whole,
    # The val results are calculated individually for each client
    # The following metrics are collected: loss, accuracy, precision,
    # confidence, f1, auc, brier, ece, positive_rate, valid_rate,
    # non-finite rate

    non_fd_params = non_fd_model.state_dict()
    torch.save(non_fd_params, output_dir / "non_fd_params.pt")
    non_fd_val_results = create_val_results_dict(non_fd_model, client_val_loaders, device)
    non_fd_test_results = create_test_results_dict(non_fd_model, test_loader, device)
    save_pickle(non_fd_val_results, output_dir / "non_fd_val_results")
    save_pickle(non_fd_test_results, output_dir / "non_fd_test_results")
    progress(
        f"finished non-federated baseline "
        f"(acc={non_fd_test_results['acc']:.4f}, auc={non_fd_test_results['auc']:.4f})"
    )

    # Train the original federated model
    progress("training original federated model")
    (
        og_model,
        og_summaries,
        og_hessian,
        og_val_results,
        trajectory,
        sum_traj,
        convergence_metrics,
        og_client_delta_rounds,
    ) = simulate_federated_learning(
        client_train_loaders,
        client_val_loaders,
        device,
        epochs=args.local_epochs,
        max_rounds=args.fl_rounds,
        lr=args.fl_lr,
        optimizer_name=args.fl_optimizer,
        weight_decay=args.fl_weight_decay,
        momentum=args.fl_momentum,
        pos_weight=args.fl_pos_weight,
        alpha=args.alpha,
        min_rounds=args.min_rounds,
        patience=args.patience,
        init_seed=args.seed + 1000,
        batch_size=args.batch_size,
        run_label="original",
        progress=True,
    )

    # Store the original federated model, curvature, summary and delta's
    og_test_results = create_test_results_dict(og_model, test_loader, device)
    save_pickle(og_val_results, output_dir / "og_val_results")
    save_pickle(og_test_results, output_dir / "og_test_results")
    og_params = og_model.state_dict()
    torch.save(og_params, output_dir / "og_params.pt")
    torch.save(og_summaries, output_dir / "og_summaries")
    torch.save(og_hessian, output_dir / "og_hessian")
    torch.save(og_client_delta_rounds, output_dir / "og_client_delta_rounds.pt")
    progress(
        f"finished original federated model "
        f"(acc={og_test_results['acc']:.4f}, auc={og_test_results['auc']:.4f})"
    )

    predictive_results = {}
    unlearning_summary = {}
    unlearning_config = {}

    # For each client, train a model without its data, and unlearn
    total_clients = len(unlearn_clients)
    for idx, client in enumerate(unlearn_clients, start=1):
        progress(f"[{idx}/{total_clients}] retraining without client {client}")
        (
            retr_model,
            _retr_summaries,
            _retr_hessian,
            _retr_train_val_results,
            _retr_trajectory,
            _retr_sum_traj,
            _retr_convergence,
            _retr_client_delta_rounds,
        ) = simulate_federated_learning(
            client_train_loaders[:client] + client_train_loaders[client + 1 :],
            client_val_loaders[:client] + client_val_loaders[client + 1 :],
            device,
            epochs=args.local_epochs,
            max_rounds=args.fl_rounds,
            lr=args.fl_lr,
            optimizer_name=args.fl_optimizer,
            weight_decay=args.fl_weight_decay,
            momentum=args.fl_momentum,
            pos_weight=args.fl_pos_weight,
            alpha=args.alpha,
            min_rounds=args.min_rounds,
            patience=args.patience,
            init_seed=args.seed + 2000 + client,
            batch_size=args.batch_size,
            run_label=f"retrain client {client}",
            progress=True,
        )

        # Store the model, as well as test and validation results
        client_path = output_dir / f"{client}"
        client_path.mkdir(parents=True, exist_ok=True)
        retr_params = retr_model.state_dict()
        torch.save(retr_params, client_path / "retr_params.pt")
        retr_val_results = create_val_results_dict(retr_model, client_val_loaders, device)
        retr_test_results = create_test_results_dict(retr_model, test_loader, device)
        save_pickle(retr_val_results, client_path / "retr_val_results")
        save_pickle(retr_test_results, client_path / "retr_test_results")
        progress(
            f"[{idx}/{total_clients}] retrain client {client} "
            f"acc={retr_test_results['acc']:.4f} auc={retr_test_results['auc']:.4f}"
        )

        # Perform unlearning for the client
        progress(f"[{idx}/{total_clients}] unlearning client {client}")
        client_eta = args.eta
        client_damping = args.damping
        client_output_scale = args.output_layer_scale
        if per_client_config is not None and client in per_client_config:
            client_eta = per_client_config[client]["eta"]
            client_damping = per_client_config[client]["damping"]
            client_output_scale = per_client_config[client].get("output_layer_scale", 1.0)
        unl_params = unlearn(
            og_params,
            og_hessian,
            og_summaries,
            client,
            args.num_clients,
            client_eta,
            client_damping,
            step_sign=args.unlearn_step_sign,
            output_layer_scale=client_output_scale,
        )
        torch.save(unl_params, client_path / "unl_params.pt")
        unl_model = build_model_from_params(unl_params, input_size)
        unl_val_results = create_val_results_dict(unl_model, client_val_loaders, device)
        unl_test_results = create_test_results_dict(unl_model, test_loader, device)
        save_pickle(unl_val_results, client_path / "unl_val_results")
        save_pickle(unl_test_results, client_path / "unl_test_results")
        unlearning_config[client] = {
            "eta": client_eta,
            "damping": client_damping,
            "step_sign": args.unlearn_step_sign,
            "output_layer_scale": client_output_scale,
        }
        progress(
            f"[{idx}/{total_clients}] unlearn client {client} "
            f"eta={client_eta:.4f} damping={client_damping:.4f} "
            f"output_scale={client_output_scale:.2f} "
            f"acc={unl_test_results['acc']:.4f} auc={unl_test_results['auc']:.4f}"
        )

        # perform two rounds of finetuning
        progress(f"[{idx}/{total_clients}] finetuning client {client}")
        (
            ft_model,
            _ft_summaries,
            _ft_hessian,
            _ft_train_val_results,
            _ft_trajectory,
            _ft_sum_traj,
            _ft_convergence,
            _ft_client_delta_rounds,
        ) = simulate_federated_learning(
            client_train_loaders[:client] + client_train_loaders[client + 1 :],
            client_val_loaders[:client] + client_val_loaders[client + 1 :],
            device,
            epochs=args.local_epochs,
            max_rounds=2,
            lr=args.fl_lr,
            optimizer_name=args.fl_optimizer,
            weight_decay=args.fl_weight_decay,
            momentum=args.fl_momentum,
            pos_weight=args.fl_pos_weight,
            alpha=args.alpha,
            min_rounds=args.min_rounds,
            patience=args.patience,
            init_seed=args.seed + 3000 + client,
            batch_size=args.batch_size,
            pretrained=unl_params,
            run_label=f"finetune client {client}",
            progress=True,
        )
        ft_params = ft_model.state_dict()
        torch.save(ft_params, client_path / "ft_params.pt")
        ft_val_results = create_val_results_dict(ft_model, client_val_loaders, device)
        ft_test_results = create_test_results_dict(ft_model, test_loader, device)
        save_pickle(ft_val_results, client_path / "ft_val_results")
        save_pickle(ft_test_results, client_path / "ft_test_results")
        progress(
            f"[{idx}/{total_clients}] finetune client {client} "
            f"acc={ft_test_results['acc']:.4f} auc={ft_test_results['auc']:.4f}"
        )

        # The predictive metrics are kl, js, tv, agreement, squared logit
        # gap, squared probability gap
        predictive_unlearn = predictive_metrics(retr_params, unl_params, test_loader, device=device)
        predictive_original_unlearn = predictive_metrics(og_params, unl_params, test_loader, device=device)
        predictive_finetune = predictive_metrics(retr_params, ft_params, test_loader, device=device)
        predictive_original_finetune = predictive_metrics(og_params, ft_params, test_loader, device=device)
        predictive_original_retrain = predictive_metrics(og_params, retr_params, test_loader, device=device)
        predictive_results[client] = {
            "unlearn_vs_retrain": predictive_unlearn,
            "original_vs_unlearn": predictive_original_unlearn,
            "finetune_vs_retrain": predictive_finetune,
            "original_vs_finetune": predictive_original_finetune,
            "original_vs_retrain": predictive_original_retrain,
        }

        # Compare the different models in terms of the aforementioned loss
        # accuracy, precision, etc.
        train_size = len(client_train_idx[client])
        empty_privacy = {}
        unlearning_summary[client] = {
            "original_vs_retrain": build_unlearning_pair_summary(
                client,
                train_size,
                "retrain",
                "original",
                retr_val_results,
                og_val_results,
                retr_test_results,
                og_test_results,
                empty_privacy,
                empty_privacy,
                predictive_original_retrain,
            ),
            "unlearn_vs_retrain": build_unlearning_pair_summary(
                client,
                train_size,
                "retrain",
                "unlearn",
                retr_val_results,
                unl_val_results,
                retr_test_results,
                unl_test_results,
                empty_privacy,
                empty_privacy,
                predictive_unlearn,
            ),
            "finetune_vs_retrain": build_unlearning_pair_summary(
                client,
                train_size,
                "retrain",
                "finetune",
                retr_val_results,
                ft_val_results,
                retr_test_results,
                ft_test_results,
                empty_privacy,
                empty_privacy,
                predictive_finetune,
            ), # what is an unlearning pair summary? 
        }

    save_pickle(predictive_results, output_dir / "predictive_results")
    save_pickle(unlearning_summary, output_dir / "unlearning_summary")
    save_json(unlearning_summary, output_dir / "unlearning_summary.json")
    save_json(unlearning_config, output_dir / "unlearning_config.json")

    experiment_summary = {
        "config": {
            "data_dir": str(args.data_dir),
            # "raw_train_path": str(raw_train_path),
            # "raw_test_path": str(raw_test_path),
            # "train_path": str(train_path),
            # "test_path": str(test_path),
            "output_dir": str(output_dir),
            "train_prop": args.train_prop,
            "seed": args.seed,
            "num_clients": args.num_clients,
            "unlearn_clients": unlearn_clients,
            "leniency": args.leniency,
            "min_client_size": args.min_client_size,
            "batch_size": args.batch_size,
            "rounds": args.rounds,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "pos_weight": args.pos_weight,
            "local_epochs": args.local_epochs,
            "fl_rounds": args.fl_rounds,
            "fl_lr": args.fl_lr,
            "fl_optimizer": args.fl_optimizer,
            "fl_weight_decay": args.fl_weight_decay,
            "fl_momentum": args.fl_momentum,
            "fl_pos_weight": args.fl_pos_weight,
            "alpha": args.alpha,
            "min_rounds": args.min_rounds,
            "patience": args.patience,
            "damping": args.damping,
            "eta": args.eta,
            "unlearn_step_sign": args.unlearn_step_sign,
            "output_layer_scale": args.output_layer_scale,
            "per_client_config_path": args.per_client_config_path,
            "input_size": input_size,
            # "kept_cluster_features": int(np.sum(kept_features)),
        },
        # "split_audit": split_audit,
        "non_federated_test": non_fd_test_results,
        "original_test": og_test_results,
        # For different model pairs, kl, js etc
        "predictive_aggregate": aggregate_predictive_results(predictive_results),
        # For different model pairs, difference in acc, prec etc
        # Take the average over the clients
        "unlearning_summary_aggregate": aggregate_pair_summaries(unlearning_summary),
    }
    save_json(experiment_summary, output_dir / "experiment_summary.json")

    progress(f"saved experiment to {output_dir}")
    progress(f"original test acc: {og_test_results['acc']:.4f}")
    progress(f"non-federated test acc: {non_fd_test_results['acc']:.4f}")


if __name__ == "__main__":
    main()
