import torch
import math
from torch.utils.data import DataLoader

from training import *
from data_utils import TabularDataset
from MIA import (
    balance_feature_rows,
    build_attack_data,
    empty_attack_metrics,
    get_features,
    mean_attack_score,
    score_attack_model,
)


def flatten_state_dict(state_dict):
    return torch.cat([value.reshape(-1) for value in state_dict.values()])


def unflatten_vector(vector, template_state):
    rebuilt = {}
    offset = 0

    for key, value in template_state.items():
        width = value.numel()
        rebuilt[key] = vector[offset:offset + width].reshape_as(value)
        offset += width

    return rebuilt


def infer_input_size_from_params(params):
    for value in params.values():
        if value.ndim == 2:
            return value.shape[1]
    raise ValueError("could not infer input size from params")


def build_client_update_basis(client_delta_rounds, client, subspace_rank):
    if client_delta_rounds is None or int(subspace_rank) <= 0:
        return None

    vectors = []
    for round_rows in client_delta_rounds:
        if client >= len(round_rows):
            continue
        vec = flatten_state_dict(round_rows[client]).float()
        if torch.isfinite(vec).all():
            vectors.append(vec)

    if not vectors:
        return None

    delta_matrix = torch.stack(vectors, dim=0)
    max_rank = min(int(subspace_rank), delta_matrix.shape[0], delta_matrix.shape[1])
    if max_rank <= 0:
        return None

    _, _, vh = torch.linalg.svd(delta_matrix, full_matrices=False)
    return vh[:max_rank].T.contiguous()


def bernoulli_kl(p, q, eps=1e-8):
    eps = max(float(eps), torch.finfo(p.dtype).eps)
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return p * (torch.log(p) - torch.log(q)) + (1 - p) * (
        torch.log1p(-p) - torch.log1p(-q)
    )


def unlearn(
    params,
    hessian,
    summaries,
    client,
    n_clients,
    eta,
    damping=None,
    step_sign=1.0,
    output_layer_scale=1.0,
    hidden_layer_scale=1.0,
    client_delta_rounds=None,
    subspace_rank=0,
    subspace_basis=None,
):
    params_vec = flatten_state_dict(params)
    summary_vec = flatten_state_dict(summaries[client]).float()
    hessian_vec = flatten_state_dict(hessian).float()
    damping_term = 0.0 if damping is None else float(damping)
    safe_hessian = torch.clamp(hessian_vec + damping_term, min=1e-8)

    param_magn = torch.abs(params_vec.float()).sum()
    summary_magn = torch.abs(summary_vec).sum().clamp_min(1e-8)
    hessian_magn = torch.abs(safe_hessian).sum()

    if eta is None:
        eta = (1 / n_clients) * (param_magn * hessian_magn) / summary_magn
        print(f"The param magn was {param_magn}, the hessian magn {hessian_magn} \
and the summmary magn {summary_magn}")
        print(f"The eta calculated by the program is {eta}")

    output_scale = float(output_layer_scale)
    hidden_scale = float(hidden_layer_scale)

    output_keys = {"fc4.weight", "fc4.bias"}
    step_state = {}
    for key in params:
        scale = output_scale if key in output_keys else hidden_scale
        layer_step = float(step_sign) * eta * summaries[client][key].float() / torch.clamp(
            hessian[key].float() + (0.0 if damping is None else float(damping)),
            min=1e-8,
        )
        step_state[key] = (scale * layer_step).type_as(params[key])

    step_vec = flatten_state_dict(step_state).float()
    basis = subspace_basis
    if basis is None and int(subspace_rank) > 0:
        basis = build_client_update_basis(client_delta_rounds, client, subspace_rank)
    if basis is not None:
        basis = basis.to(step_vec)
        step_vec = basis @ (basis.T @ step_vec)

    return unflatten_vector(params_vec.float() + step_vec, params)


def predictive_metrics(reference_params, candidate_params, test_loader, device):
    input_size = infer_input_size_from_params(reference_params)
    retrained_model = Net(input_size=input_size)
    retrained_model.load_state_dict(reference_params)
    retrained_model.to(device)
    retrained_model.eval()

    unlearned_model = Net(input_size=input_size)
    unlearned_model.load_state_dict(candidate_params)
    unlearned_model.to(device)
    unlearned_model.eval()

    kl = 0.0
    js = 0.0
    tv = 0.0
    agreement = 0.0
    sample_size = 0
    valid_size = 0
    squared_logit_gap = 0.0
    squared_prob_gap = 0.0

    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)

            logit_retr = retrained_model(X)
            logit_unl = unlearned_model(X)
            valid_mask = torch.isfinite(logit_retr) & torch.isfinite(logit_unl)

            sample_size += len(X)
            if not valid_mask.any():
                continue

            logit_retr = logit_retr[valid_mask]
            logit_unl = logit_unl[valid_mask]
            valid_size += int(valid_mask.sum().item())

            p = torch.sigmoid(logit_retr)
            q = torch.sigmoid(logit_unl)
            m = 0.5 * (p + q)
            batch_kl = bernoulli_kl(p, q)
            batch_js = 0.5 * (bernoulli_kl(p, m) + bernoulli_kl(q, m))

            kl += float(batch_kl.sum().item())
            js += float(batch_js.sum().item())
            tv += float(torch.abs(p - q).sum().item())
            agreement += float(((p > 0.5) == (q > 0.5)).float().sum().item())
            squared_logit_gap += float(((logit_retr - logit_unl) ** 2).sum().item())
            squared_prob_gap += float(((p - q) ** 2).sum().item())

    if valid_size == 0:
        return {
            "kl": float("nan"),
            "js": float("nan"),
            "tv": float("nan"),
            "agreement": float("nan"),
            "logit_l2": float("nan"),
            "logit_rmse": float("nan"),
            "prob_rmse": float("nan"),
            "valid_rate": 0.0,
            "non_finite_rate": 1.0,
            "n_examples": sample_size,
        }

    return {
        "kl": kl / valid_size,
        "js": js / valid_size,
        "tv": tv / valid_size,
        "agreement": agreement / valid_size,
        "logit_l2": math.sqrt(squared_logit_gap) / valid_size,
        "logit_rmse": math.sqrt(squared_logit_gap / valid_size),
        "prob_rmse": math.sqrt(squared_prob_gap / valid_size),
        "valid_rate": valid_size / sample_size,
        "non_finite_rate": 1 - (valid_size / sample_size),
        "n_examples": sample_size,
    }


def predictive_kl_and_l2(reference_params, candidate_params, test_loader, device):
    metrics = predictive_metrics(reference_params, candidate_params, test_loader, device)
    return metrics["kl"], metrics["logit_l2"]

def membership_inference_attack(
    model_params,
    non_members,
    members,
    attack_model,
    device,
    focus_loader=None,
):

    input_size = infer_input_size_from_params(model_params)
    model = Net(input_size=input_size)
    model.load_state_dict(model_params)
    
    non_member_features = get_features(model, non_members, device)
    member_features = get_features(model, members, device)
    member_features, non_member_features = balance_feature_rows(
        member_features,
        non_member_features,
    )

    if member_features.numel() == 0 or non_member_features.numel() == 0:
        metrics = empty_attack_metrics()
        if focus_loader is not None:
            metrics["forgotten_score"] = float("nan")
        return metrics

    member_data, non_member_data = build_attack_data(member_features, non_member_features)
    data = torch.cat((member_data, non_member_data), dim=0)
    data = TabularDataset(data[:, :-1], data[:, -1])
    data_loader = DataLoader(data, batch_size=32, shuffle=False)
    metrics = score_attack_model(attack_model, data_loader, device)

    if focus_loader is not None:
        focus_features = get_features(model, focus_loader, device)
        metrics["forgotten_score"] = mean_attack_score(attack_model, focus_features, device)

    return metrics
