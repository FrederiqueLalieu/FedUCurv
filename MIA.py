import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from training import *
from data_utils import TabularDataset


class MIA_small(nn.Module):
    def __init__(self, input_size=5):
        super(MIA_small, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)
        self.fc2 = nn.Linear(4, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MIA_large(nn.Module):
    def __init__(self, input_size=5):
        super(MIA_large, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        return self.fc3(x)


def empty_attack_metrics():
    return {
        "loss": float("nan"),
        "acc": float("nan"),
        "auc": float("nan"),
        "balanced_acc": float("nan"),
        "tpr": float("nan"),
        "tnr": float("nan"),
        "member_score": float("nan"),
        "non_member_score": float("nan"),
    }

def performance_metrics(probs, y):    
    preds = (probs > 0.5).float()
    correct += int((preds == y).sum().item())
    predicted_positive = preds == 1
    actual_positive = y == 1
    tp += int((predicted_positive & actual_positive).sum().item())
    pred_pos += int(predicted_positive.sum().item())
    actual_pos += int(actual_positive.sum().item())

    acc = correct / len(preds)
    prec = float(tp / (pred_pos + 1e-8))
    recall = float(tp / (actual_pos + 1e-8))

    return acc, prec, recall

def get_features(net, dataloader, device):
    features = None
    net.to(device)
    net.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).float()
            logit = net(X).squeeze(-1)

            p = torch.sigmoid(logit)
            loss = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            preds = (p > 0.5).float()
            correct = (preds.squeeze() == y).float()
            inv_p = torch.ones_like(p) - p
            confidence = torch.max(p, inv_p)
            entropy = -p * torch.log(p + 1e-8) - inv_p * torch.log(inv_p + 1e-8)

            batch_features = torch.hstack(
                [
                    logit.unsqueeze(1),
                    loss.unsqueeze(1),
                    correct.unsqueeze(1),
                    confidence.unsqueeze(1),
                    entropy.unsqueeze(1),
                ]
            )

            if features is None:
                features = batch_features
            else:
                features = torch.vstack((features, batch_features))

    return features


def finite_feature_rows(features):
    if features is None or features.numel() == 0:
        return features

    row_mask = torch.isfinite(features).all(dim=1)
    return features[row_mask]


def balance_feature_rows(member_features, non_member_features, seed=0):
    """" Makes the number of samples in each category equal. 
    """
    member_features = finite_feature_rows(member_features)
    non_member_features = finite_feature_rows(non_member_features)

    sample_size = min(member_features.size(0), non_member_features.size(0))
    generator = torch.Generator()
    generator.manual_seed(seed)

    member_pick = torch.randperm(member_features.size(0), generator=generator)[:sample_size]
    non_member_pick = torch.randperm(non_member_features.size(0), generator=generator)[:sample_size]

    return member_features[member_pick], non_member_features[non_member_pick]


def build_attack_data(member_features, non_member_features):
    zeros_row = torch.zeros((non_member_features.size(0), 1))
    ones_row = torch.ones((member_features.size(0), 1))

    non_member_data = torch.cat((non_member_features, zeros_row), dim=1)
    member_data = torch.cat((member_features, ones_row), dim=1)

    return member_data, non_member_data


def score_attack_model(attack_model, data_loader, device):
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    attack_model.to(device)
    attack_model.eval()

    logits_all = []
    labels_all = []
    total_loss = 0.0

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device).float()
            logits = attack_model(X).squeeze(-1)
            total_loss += float(criterion(logits, y).item())
            logits_all.append(logits.cpu())
            labels_all.append(y.cpu())

    if not logits_all:
        return empty_attack_metrics()

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    finite_mask = torch.isfinite(logits) & torch.isfinite(labels)
    logits = logits[finite_mask]
    labels = labels[finite_mask]

    if labels.numel() == 0:
        return empty_attack_metrics()

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    member_mask = labels == 1
    non_member_mask = labels == 0
    tpr = float("nan") if not torch.any(member_mask) else float((preds[member_mask] == 1).float().mean().item())
    tnr = float("nan") if not torch.any(non_member_mask) else float((preds[non_member_mask] == 0).float().mean().item())
    if torch.unique(labels).numel() < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels.numpy(), probs.numpy()))

    acc, prec, recall = performance_metrics(probs, labels)

    metrics = {
        "loss": total_loss / len(labels),
        "acc" : acc,
        "recall" : recall,
        "prec" : prec,
        "acc": float((preds == labels).float().mean().item()),
        "auc": auc,
        "balanced_acc": 0.5 * (tpr + tnr),
        "tpr": tpr,
        "tnr": tnr,
        "member_score": float("nan") if not torch.any(member_mask) else float(probs[member_mask].mean().item()),
        "non_member_score": float("nan") if not torch.any(non_member_mask) else float(probs[non_member_mask].mean().item()),
    }

    return metrics


def predict_attack_probs(attack_model, features, device):
    features = finite_feature_rows(features)
    if features is None or features.numel() == 0:
        return torch.tensor([])

    attack_model.to(device)
    attack_model.eval()

    with torch.no_grad():
        logits = attack_model(features.to(device)).squeeze(-1)
        probs = torch.sigmoid(logits)

    return probs.cpu()


def mean_attack_score(attack_model, features, device):
    probs = predict_attack_probs(attack_model, features, device)
    if probs.numel() == 0:
        return float("nan")
    return float(probs.mean().item())


def train_MIA(model, used_during_training, not_used_during_training, device, large=True):
    member_features = get_features(model, used_during_training, device)
    non_member_features = get_features(model, not_used_during_training, device)
    member_features, non_member_features = balance_feature_rows(
        member_features,
        non_member_features,
    )

    # split the data into testing and training
    split_point = int(0.8 * member_features.shape[0])
    member_train = member_features[:split_point]
    member_test = member_features[split_point:]
    non_member_train = non_member_features[:split_point]
    non_member_test = non_member_features[split_point:]

    # add the labels to train, split to X and y
    member_train_data, non_member_train_data = build_attack_data(member_train, non_member_train)
    train_data = torch.cat((member_train_data, non_member_train_data), dim=0)
    x_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    trainloader = DataLoader(TabularDataset(x_train, y_train), batch_size=32, shuffle=True)

    # train the MIA model
    if large is True:
        attack_model = MIA_large(input_size=x_train.shape[1])
    else:
        attack_model = MIA_small(input_size=x_train.shape[1])

    train_one_model(attack_model, trainloader, device, 30, 0.01)

    # evaluate the model
    member_test_data, non_member_test_data = build_attack_data(member_test, non_member_test)
    test_data = torch.cat((member_test_data, non_member_test_data), dim=0)
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    testloader = DataLoader(TabularDataset(x_test, y_test), batch_size=32, shuffle=False)
    metrics = score_attack_model(attack_model, testloader, device)

    return attack_model, metrics


def get_client_bag_features(net, dataloader, device):
    """ Returns the features MIA uses for a complete batch. Given a dataloader
    with M batches, each containing N samples, it will return a tensor of
    size (M*N) x 6.
    """
    rows = []
    net.to(device)
    net.eval()

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).float()

            h1 = F.relu(net.fc1(X))
            h2 = F.relu(net.fc2(h1))
            h3 = F.relu(net.fc3(h2))
            logit = net.fc4(h3).squeeze(-1)

            p = torch.sigmoid(logit)
            inv_p = torch.ones_like(p) - p
            loss = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            confidence = torch.maximum(p, inv_p)
            entropy = -p * torch.log(p + 1e-8) - inv_p * torch.log(inv_p + 1e-8)
            true_prob = y * p + (1 - y) * inv_p

            batch_features = torch.cat(
                [
                    logit.unsqueeze(1),
                    loss.unsqueeze(1),
                    confidence.unsqueeze(1),
                    entropy.unsqueeze(1),
                    true_prob.unsqueeze(1),
                    h3,
                ],
                dim=1,
            )

            # remove the samples that contains an infinite value
            finite_mask = torch.isfinite(batch_features).all(dim=1)
            if torch.any(finite_mask):
                rows.append(batch_features[finite_mask].cpu().numpy())
    
    # if all samples contianed infinite values
    if not rows:
        return np.zeros((0, 10), dtype=np.float32) # why the 10?

    return np.concatenate(rows, axis=0)


def sample_bag_summaries(feature_rows, n_bags=30, bag_size=128, seed=0):
    """ Samples from a tensor with MIA features, a speicified number of
    bags, a specified number of entries. For each bag and each feature,
    the mean, average and quantiles of each MIA feature is returned. 
    So given a tensor of size NxM, the result will be n_bags x (5*M).
    """
    if feature_rows.shape[0] == 0:
        return np.zeros((0, feature_rows.shape[1] * 5), dtype=np.float32) # why the 5?

    rng = np.random.default_rng(seed)
    rows = []
    bag_width = min(int(bag_size), feature_rows.shape[0])

    for _ in range(int(n_bags)):
        picked = rng.choice(feature_rows.shape[0], size=bag_width, replace=False)
        bag = feature_rows[picked]
        rows.append(
            np.concatenate(
                [
                    bag.mean(axis=0),
                    bag.std(axis=0),
                    np.quantile(bag, 0.25, axis=0),
                    np.quantile(bag, 0.5, axis=0),
                    np.quantile(bag, 0.75, axis=0),
                ]
            )
        )

    return np.asarray(rows, dtype=np.float32)


def train_client_bag_attack(
    shadow_model,
    member_loaders,
    non_member_loaders,
    device,
    bags_per_client=30,
    bag_size=128,
    seed=0,
):
    """For each bag, get the features, summarize, add the labels. Concatenate all
    the bags and train a MIA model on it.
    """
    X_rows = []
    y_rows = []

    for idx, (member_loader, non_member_loader) in enumerate(zip(member_loaders, non_member_loaders)):
        member_features = get_client_bag_features(shadow_model, member_loader, device)
        non_member_features = get_client_bag_features(shadow_model, non_member_loader, device)
        member_bags = sample_bag_summaries(
            member_features,
            n_bags=bags_per_client,
            bag_size=bag_size,
            seed=seed + idx,
        )
        non_member_bags = sample_bag_summaries(
            non_member_features,
            n_bags=bags_per_client,
            bag_size=bag_size,
            seed=seed + 1000 + idx,
        )

        if member_bags.shape[0] == 0 or non_member_bags.shape[0] == 0:
            continue

        X_rows.extend([member_bags, non_member_bags])
        y_rows.extend(
            [
                np.ones(member_bags.shape[0], dtype=np.float32),
                np.zeros(non_member_bags.shape[0], dtype=np.float32),
            ]
        )

    if not X_rows:
        return None, {"auc": float("nan"), "acc": float("nan"), "prec": float("nan"), "recall": float("nan"), "n_clients": 0, "bags_per_client": bags_per_client, "bag_size": bag_size}

    X_train = np.concatenate(X_rows, axis=0)
    y_train = np.concatenate(y_rows, axis=0)
    attack_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    attack_model.fit(X_train, y_train) # what is the difference between this and train one model?
    probs = attack_model.predict_proba(X_train)[:, 1]
    auc = float(roc_auc_score(y_train, probs))
    acc, prec, recall = performance_metrics(probs, y_train)

    return attack_model, {
        "auc": auc,
        "acc" : acc,
        "prec" : prec,
        "recall" : recall,
        "n_clients": len(member_loaders),
        "bags_per_client": bags_per_client,
        "bag_size": bag_size,
    }


def score_client_bag_attack(
    attack_model,
    target_model,
    member_loader,
    non_member_loader,
    device,
    eval_bags=20,
    bag_size=128,
    seed=0,
):
    """Given a target model and member and non-member data points, gives the auc
    of the attack model, the average prob of the member and non-member group 
    and the difference.
    """
    if attack_model is None:
        return {
            "auc": float("nan"),
            "member_score": float("nan"),
            "non_member_score": float("nan"),
            "score_gap": float("nan"),
        }

    member_features = get_client_bag_features(target_model, member_loader, device)
    non_member_features = get_client_bag_features(target_model, non_member_loader, device)
    member_bags = sample_bag_summaries(member_features, n_bags=eval_bags, bag_size=bag_size, seed=seed)
    non_member_bags = sample_bag_summaries(non_member_features, n_bags=eval_bags, bag_size=bag_size, seed=seed + 1000)

    if member_bags.shape[0] == 0 or non_member_bags.shape[0] == 0:
        return {
            "auc": float("nan"),
            "member_score": float("nan"),
            "non_member_score": float("nan"),
            "score_gap": float("nan"),
        }

    X_eval = np.concatenate([member_bags, non_member_bags], axis=0)
    y_eval = np.concatenate(
        [
            np.ones(member_bags.shape[0], dtype=np.float32),
            np.zeros(non_member_bags.shape[0], dtype=np.float32),
        ]
    )
    probs = attack_model.predict_proba(X_eval)[:, 1]
    member_scores = probs[: member_bags.shape[0]]
    non_member_scores = probs[member_bags.shape[0] :]

    acc, prec, recall = performance_metrics(probs, y_eval)

    return {
        "acc" : acc, 
        "prec" : prec, 
        "recall" : recall,
        "auc": float(roc_auc_score(y_eval, probs)),
        "member_score": float(member_scores.mean()),
        "non_member_score": float(non_member_scores.mean()),
        "score_gap": float(member_scores.mean() - non_member_scores.mean()),
    }


def build_client_feature_rows(model, client_loaders, device):
    return [get_client_bag_features(model, loader, device) for loader in client_loaders]


def sample_client_delta_bags(reference_rows, candidate_rows, n_bags=12, bag_size=256, seed=0):
    """ Sample a specified number of bag pairs (one from reference, one from
    candidate). For each pair, calculate the difference for each feature, and 
    then for each the mean, std, also for the absolute and quantiles. So if you 
    have N1 reference rows, N2 candidate rows and M features, you will get a 
    tensor with shape n_bags x (M*7).
    """
    if reference_rows.shape[0] == 0 or candidate_rows.shape[0] == 0:
        width = reference_rows.shape[1] if reference_rows.shape[0] else candidate_rows.shape[1]
        return np.zeros((0, width * 7), dtype=np.float32)

    n = min(reference_rows.shape[0], candidate_rows.shape[0])
    reference_rows = reference_rows[:n]
    candidate_rows = candidate_rows[:n]
    rng = np.random.default_rng(seed)
    bag_width = min(int(bag_size), n)
    rows = []

    for _ in range(int(n_bags)):
        picked = rng.choice(n, size=bag_width, replace=False)
        delta = candidate_rows[picked] - reference_rows[picked]
        abs_delta = np.abs(delta)
        rows.append(
            np.concatenate(
                [
                    delta.mean(axis=0),
                    delta.std(axis=0),
                    np.quantile(delta, 0.25, axis=0),
                    np.quantile(delta, 0.5, axis=0),
                    np.quantile(delta, 0.75, axis=0),
                    abs_delta.mean(axis=0),
                    abs_delta.std(axis=0),
                ]
            )
        )

    return np.asarray(rows, dtype=np.float32)


def train_client_deletion_attack(
    original_rows,
    retrain_rows,
    holdout_client,
    train_bags=12,
    cross_negative_bags=4,
    bag_size=256,
    attack_c=1.0,
    seed=0,
):
    X_rows = []
    y_rows = []
    clients = [idx for idx in range(len(original_rows)) if idx != holdout_client]

    for client in clients:
        # Retrieve delta info for original-retrain
        positive_rows = sample_client_delta_bags(
            original_rows[client],
            retrain_rows[client][client],
            n_bags=train_bags,
            bag_size=bag_size,
            seed=int(seed) + 1000 + holdout_client * 100 + client,
        )
        original_control = sample_client_delta_bags(
            original_rows[client], # wait but delta is always 0 here right?
            original_rows[client],
            n_bags=train_bags,
            bag_size=bag_size,
            seed=int(seed) + 2000 + holdout_client * 100 + client,
        )
        X_rows.extend([positive_rows, original_control])
        y_rows.extend(
            [
                np.ones(positive_rows.shape[0], dtype=np.float32),
                np.zeros(original_control.shape[0], dtype=np.float32),
            ]
        )

        for removed_client in clients:
            if removed_client == client:
                continue

            negative_rows = sample_client_delta_bags(
                original_rows[client],
                retrain_rows[removed_client][client],
                n_bags=cross_negative_bags,
                bag_size=min(int(bag_size // 2), original_rows[client].shape[0], retrain_rows[removed_client][client].shape[0]),
                seed=int(seed) + 3000 + holdout_client * 100 + client * 10 + removed_client,
            )
            X_rows.append(negative_rows)
            y_rows.append(np.zeros(negative_rows.shape[0], dtype=np.float32))

    X_train = np.concatenate(X_rows, axis=0)
    y_train = np.concatenate(y_rows, axis=0)
    attack_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=float(attack_c),
        ),
    )
    attack_model.fit(X_train, y_train)
    probs = attack_model.predict_proba(X_train)[:, 1]
    train_auc = float(roc_auc_score(y_train, probs))
    
    train_acc, train_prec, train_recall = performance_metrics(probs, y_train)

    return attack_model, {
        "train_auc": train_auc,
        "train_acc": train_acc,
        "train_prec": train_prec,
        "train_recall": train_recall,
        "n_train_rows": int(X_train.shape[0]),
        "holdout_client": int(holdout_client),
        "attack_c": float(attack_c),
        "seed": int(seed),
    }


def score_client_deletion_attack(
    attack_model,
    original_rows,
    candidate_rows,
    target_client,
    eval_bags=10,
    bag_size=256,
    seed=0,
):
    """ The attacker tries to predict whether a client was deleted based on the
    difference in model features before and after deleting.
    """
    scores = []
    labels = []
    retained_bag_scores = []
    target_bag_scores = None

    for client in range(len(original_rows)):
        bag_rows = sample_client_delta_bags(
            original_rows[client],
            candidate_rows[client],
            n_bags=eval_bags,
            bag_size=min(int(bag_size), original_rows[client].shape[0], candidate_rows[client].shape[0]),
            seed=seed + target_client * 100 + client,
        )
        bag_scores = attack_model.predict_proba(bag_rows)[:, 1]
        score = float(bag_scores.mean())
        scores.append(score)
        labels.append(1 if client == target_client else 0)
        if client == target_client:
            target_bag_scores = bag_scores
        else:
            retained_bag_scores.append(bag_scores)

    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    retained_scores = scores[labels == 0]
    retained_bag_scores = np.concatenate(retained_bag_scores, axis=0)
    target_bag_scores = np.asarray(target_bag_scores, dtype=np.float32)
    rank = int(np.where(np.argsort(-scores) == target_client)[0][0]) + 1
    bag_labels = np.concatenate(
        [
            np.ones(target_bag_scores.shape[0], dtype=np.float32),
            np.zeros(retained_bag_scores.shape[0], dtype=np.float32),
        ]
    )

    train_acc, train_prec, train_recall = performance_metrics(bag_scores, bag_labels)


    bag_scores = np.concatenate([target_bag_scores, retained_bag_scores], axis=0)
    bag_auc = float(roc_auc_score(bag_labels, bag_scores))
    client_auc = float(roc_auc_score(labels, scores))
    bag_deleted_score = float(target_bag_scores.mean())
    bag_retained_mean = float(retained_bag_scores.mean())
    bag_score_gap = float(bag_deleted_score - bag_retained_mean)
    bag_score_z = float(bag_score_gap / (float(retained_bag_scores.std()) + 1e-8))
    bag_quantile = float((retained_bag_scores < bag_deleted_score).mean())

    return {
        "acc" : train_acc,
        "prec" : train_prec,
        "recall" : train_recall,
        "auc": bag_auc,
        "client_auc": client_auc,
        "deleted_score": bag_deleted_score,
        "retained_mean_score": bag_retained_mean,
        "score_gap": bag_score_gap,
        "score_z": bag_score_z,
        "quantile": bag_quantile,
        "quantile_gap": float(bag_quantile - 0.5),
        "client_deleted_score": float(scores[target_client]),
        "client_retained_mean_score": float(retained_scores.mean()),
        "client_score_gap": float(scores[target_client] - retained_scores.mean()),
        "rank": rank,
        "scores": scores.tolist(),
    }
