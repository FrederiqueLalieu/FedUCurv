import math


def finite_gap(candidate, baseline):
    candidate = float(candidate)
    baseline = float(baseline)

    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return {"gap": float("nan"), "abs_gap": float("nan")}

    gap = candidate - baseline
    return {"gap": gap, "abs_gap": abs(gap)}


def mean_without(values, skip_idx):
    kept = []

    for idx, value in enumerate(values):
        if idx == skip_idx:
            continue

        value = float(value)
        if math.isfinite(value):
            kept.append(value)

    if not kept:
        return float("nan")

    return sum(kept) / len(kept)


def shared_keys(left, right):
    return sorted(set(left.keys()) & set(right.keys()))


def scalar_metric_gaps(baseline_metrics, candidate_metrics):
    """ Independent of client to be deleted, calculate for each metric the 
    difference between baseline and candidate performance. 
    """
    summary = {}

    for metric in shared_keys(baseline_metrics, candidate_metrics):
        baseline_value = baseline_metrics[metric]
        candidate_value = candidate_metrics[metric]

        if isinstance(baseline_value, (list, tuple)) or isinstance(candidate_value, (list, tuple)):
            continue

        summary[metric] = finite_gap(candidate_value, baseline_value)

    return summary


def val_metric_gaps(baseline_results, candidate_results, forgotten_client):
    """ Given a client to be forgotten, calculate 1) the difference between the
    baseline on this client and the candidate on this client and 2) the 
    difference between the baseline on all the other clients and the candidate 
    on all the other clients.
    """
    forgotten_summary = {}
    retained_summary = {}

    for metric in shared_keys(baseline_results, candidate_results):
        baseline_values = baseline_results[metric]
        candidate_values = candidate_results[metric]

        if not isinstance(baseline_values, (list, tuple)) or not isinstance(candidate_values, (list, tuple)):
            continue

        forgotten_summary[metric] = finite_gap(
            candidate_values[forgotten_client],
            baseline_values[forgotten_client],
        )
        retained_summary[metric] = finite_gap(
            mean_without(candidate_values, forgotten_client),
            mean_without(baseline_values, forgotten_client),
        )

    return {
        "forgotten_client": forgotten_summary,
        "retained_clients_mean": retained_summary,
    }


def build_unlearning_pair_summary(
    client,
    train_size,
    baseline_name,
    candidate_name,
    baseline_val_results,
    candidate_val_results,
    baseline_test_results,
    candidate_test_results,
    baseline_privacy_metrics,
    candidate_privacy_metrics,
    predictive_metrics,
):
    return {
        "client": int(client),
        "train_size": int(train_size),
        "baseline": baseline_name,
        "candidate": candidate_name,
        "predictive": predictive_metrics,
        "utility": {
            "val": val_metric_gaps(baseline_val_results, candidate_val_results, client),
            "test": scalar_metric_gaps(baseline_test_results, candidate_test_results),
        },
        "privacy": scalar_metric_gaps(baseline_privacy_metrics, candidate_privacy_metrics),
    }
