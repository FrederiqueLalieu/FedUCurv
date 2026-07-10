import argparse
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset


TARGET_COLUMN = "was_blocked"
CATEGORICAL_FEATURES = [
    "Company",
    "Item Type",
    "Spend area text",
    "Sub spend area text",
    "Source",
    "Item Category",
    "Document Type",
]
NUMERIC_FEATURES = [
    "GR-Based Inv. Verif.",
    "Goods Receipt",
    "n_events",
    "n_invoices",
    "n_goods_receipts",
    "n_clearing_events",
    "order_value",
    "sum_invoices",
    "log_ratio",
    "case_duration",
    "goods_till_invoice_missing",
    "goods_till_invoice",
    "goods_till_clear_missing",
    "goods_till_clear",
    "has_goods_receipt",
    "has_invoice_receipt",
    "has_clear_invoice",
]
IGNORED_COLUMNS = [
    "case_id",
    "Purchasing Document",
    "Item",
    "Vendor",
    "Name",
    "blocked_event_count",
    "item_block_event_count",
]


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def display_results(results):
    for metric, result in results.items():
        print("\n" + metric)
        for client in result:
            print(f"{client:.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", default="raw_train_data.pkl")
    parser.add_argument("--raw-test", default="raw_test_data.pkl")
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def load_case_frame(file_path: str | Path):
    return pd.read_pickle(file_path)


def split_features_and_target(df):
    X = df.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    return X, y


def load_feature_metadata(metadata_path="feature_metadata.json"):
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pkl_as_dataset(file_path: str, metadata_path=None, artificial=False):
    if artificial:
        data = pd.read_pickle(file_path)
        print(data)
        dataset = TabularDataset(data['x'], data['y'])
        metadata = {'input_size': data['x'].shape[1]}
    else:
        file_path = Path(file_path)
        if metadata_path is None:
            metadata_path = file_path.parent / "feature_metadata.json"

        df = pd.read_pickle(file_path)
        print(df)
        metadata = load_feature_metadata(metadata_path)
        feature_df = df[metadata["feature_names"] + [TARGET_COLUMN]]
        X, y = split_features_and_target(feature_df)
        dataset = TabularDataset(X, y)
    return dataset, metadata


def prepare_feature_frame(df):
    feature_df = df.copy()
    for column in CATEGORICAL_FEATURES:
        feature_df[column] = feature_df[column].fillna("").astype(str)
    for column in NUMERIC_FEATURES:
        feature_df[column] = feature_df[column].fillna(0.0).astype(float)
    return feature_df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_case_frames(raw_train_df, raw_test_df):
    train_features = prepare_feature_frame(raw_train_df)
    test_features = prepare_feature_frame(raw_test_df)

    preprocessor = build_preprocessor()
    train_array = preprocessor.fit_transform(train_features)
    test_array = preprocessor.transform(test_features)
    feature_names = preprocessor.get_feature_names_out().tolist()

    train_df = pd.DataFrame(train_array, columns=feature_names)
    train_df[TARGET_COLUMN] = raw_train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    test_df = pd.DataFrame(test_array, columns=feature_names)
    test_df[TARGET_COLUMN] = raw_test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    metadata = {
        "target_column": TARGET_COLUMN,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "ignored_columns": IGNORED_COLUMNS,
        "feature_names": feature_names,
        "input_size": len(feature_names),
    }

    return train_df, test_df, metadata, preprocessor


def save_processed_artifacts(
    raw_train_path="raw_train_data.pkl",
    raw_test_path="raw_test_data.pkl",
    output_dir=".",
):
    """ Prepares the data to be used for training and testing. It scales
    the numerical features, applies one-hot encoding to the categorical
    features and removes the ignored columns. It then saves the data
    frames, information about the features and the processor.
    """
    output_dir = Path(output_dir)
    raw_train_df = load_case_frame(raw_train_path)
    raw_test_df = load_case_frame(raw_test_path)

    train_df, test_df, metadata, preprocessor = transform_case_frames(
        raw_train_df,
        raw_test_df,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_pickle(output_dir / "train_data.pkl")
    test_df.to_pickle(output_dir / "test_data.pkl")

    with open(output_dir / "feature_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    with open(output_dir / "preprocessor.pkl", "wb") as handle:
        pickle.dump(preprocessor, handle)

    return train_df, test_df, metadata


def get_cluster_matrix(dataset, std_floor=1e-2):
    """ Filters out the columns with std under the threshold std_floor.
    """
    X = np.asarray(dataset.X, dtype=np.float32)
    std = X.std(axis=0)
    keep_mask = std > std_floor
    if not np.any(keep_mask):
        raise ValueError("no usable columns left for clustering")
    return X[:, keep_mask], keep_mask


def top_up_small_clusters(Z, centers, labels, min_size):
    """ Redistributes data points over the clients so that their sizes 
    are more even. 
    """
    labels = labels.copy()
    counts = np.bincount(labels, minlength=centers.shape[0])

    while counts.min() < min_size:
        small = int(np.argmin(counts))
        donors = np.where(counts > min_size)[0]
        if donors.size == 0:
            raise ValueError("could not reach the min client size")

        donor = int(donors[np.argmax(counts[donors])])
        donor_idx = np.where(labels == donor)[0]

        current = ((Z[donor_idx] - centers[donor]) ** 2).sum(axis=1)
        target = ((Z[donor_idx] - centers[small]) ** 2).sum(axis=1)
        move_idx = donor_idx[np.argmin(target - current)]

        labels[move_idx] = small
        counts[donor] -= 1
        counts[small] += 1

    return labels


# ---------- GLOBAL TRAIN / VAL SPLIT ----------------------------------


def global_train_val_split(dataset, train_prop=0.8, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    split = int(train_prop * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    return train_ds, val_ds, train_idx, val_idx


def make_dataloaders(train_ds, val_ds, batch_size: int, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, generator=g)
    return train_loader, val_loader


# ---------- FEDERATED SPLITTING ---------------------------------------


def dirichlet_partitions(y, num_clients, alpha=100, seed=42):
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(y))
    class_indices = [np.where(y == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for indices in class_indices:
        rng.shuffle(indices)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        counts = [max((proportion * len(indices)).astype(int), 1) for proportion in proportions]

        diff = len(indices) - np.sum(counts)
        for i in range(diff):
            counts[i % num_clients] += 1

        start = 0
        for i in range(num_clients):
            end = start + counts[i]
            client_indices[i].extend(indices[start:end])
            start = end

    return [np.array(idx, dtype=int) for idx in client_indices]


def balance_clusters(Z, centers, labels, leniency, min_size=0):
    N, _ = Z.shape
    K = centers.shape[0]

    D = Z[:, None, :] - centers[None, :, :]
    D = (D**2).sum(axis=2)

    pref = np.argsort(D, axis=1)

    capacity = N // K
    cap_low = max(min_size, int(np.floor(capacity * (1 - leniency))))
    cap_high = int(np.ceil(capacity * (1 + leniency)))
    counts = np.array([np.sum(labels == k) for k in range(K)], dtype=int)

    for p in range(1, K):
        p_best = pref[:, p]
        p_margin = D[np.arange(N), p_best] - D[np.arange(N), pref[:, 0]]
        p_order = np.argsort(p_margin)

        for i in p_order:
            old = labels[i]
            new = p_best[i]

            if counts[new] >= cap_high:
                continue
            if counts[old] <= cap_low:
                continue

            labels[i] = new
            counts[old] -= 1
            counts[new] += 1

    return top_up_small_clusters(Z, centers, labels, min_size)


def balanced_feature_clusters(
    data,
    num_clients,
    seed=42,
    leniency=0.1,
    min_client_size=1000,
    std_floor=1e-2,
):
    X, keep_mask = get_cluster_matrix(data, std_floor=std_floor)

    pca_dims = min(5, X.shape[1], X.shape[0])
    pca = PCA(n_components=pca_dims, random_state=seed)
    Z = pca.fit_transform(X)

    km = KMeans(n_clusters=num_clients, random_state=seed, n_init="auto")
    km.fit(Z)
    centers = km.cluster_centers_
    labels = km.labels_

    labels = np.array(
        balance_clusters(
            Z,
            centers,
            labels,
            leniency=leniency,
            min_size=min_client_size,
        )
    )
    labels_arr = np.array(labels)
    client_indices = [np.where(labels_arr == client)[0] for client in range(num_clients)]

    return client_indices, keep_mask


def create_clients(data, num_clients, seed=42, leniency=0.1, min_client_size=1000):
    client_indices, _ = balanced_feature_clusters(
        data,
        num_clients=num_clients,
        seed=seed,
        leniency=leniency,
        min_client_size=min_client_size,
    )
    return client_indices


def client_train_val_split(client_indices, train_prop=0.8, seed=42, labels=None):
    rng = np.random.default_rng(seed)
    train_splits, val_splits = [], []

    for indices in client_indices:
        idx = indices.copy()
        rng.shuffle(idx)
        if labels is not None:
            client_labels = np.asarray(labels)[idx]
            label_counts = np.bincount(client_labels.astype(int), minlength=2)
            can_stratify = (
                np.count_nonzero(label_counts) > 1
                and label_counts.min() >= 2
                and len(idx) >= 4
            )
            if can_stratify:
                train_idx, val_idx = train_test_split(
                    idx,
                    train_size=train_prop,
                    random_state=seed,
                    shuffle=True,
                    stratify=client_labels,
                )
                train_splits.append(np.array(train_idx, dtype=int))
                val_splits.append(np.array(val_idx, dtype=int))
                continue

        split = int(train_prop * len(idx))
        train_splits.append(np.array(idx[:split], dtype=int))
        val_splits.append(np.array(idx[split:], dtype=int))

    return train_splits, val_splits


def make_federated_loaders(dataset, client_train_idx, client_val_idx, batch_size, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)

    num_clients = len(client_train_idx)
    client_train_loaders = []
    client_val_loaders = []

    for i in range(num_clients):
        train_ds = Subset(dataset, client_train_idx[i])
        val_ds = Subset(dataset, client_val_idx[i])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, generator=g)

        client_train_loaders.append(train_loader)
        client_val_loaders.append(val_loader)

    return client_train_loaders, client_val_loaders


def make_global_loader_from_clients(dataset, client_val_idx, batch_size, seed=42, shuffle=False):
    g = torch.Generator()
    g.manual_seed(seed)

    all_val_idx = np.concatenate(client_val_idx)
    val_ds = Subset(dataset, all_val_idx)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle, generator=g)
    return val_loader, all_val_idx


def build_split_audit(raw_train_df, client_indices):
    client_stats = []
    blocked_rates = []

    for client_id, indices in enumerate(client_indices):
        client_df = raw_train_df.iloc[indices]
        blocked_rate = float(client_df[TARGET_COLUMN].mean())
        blocked_rates.append(blocked_rate)
        client_stats.append(
            {
                "client": client_id,
                "size": int(len(client_df)),
                "blocked_rate": blocked_rate,
                "top_spend_areas": client_df["Spend area text"].value_counts().head(3).to_dict(),
                "top_item_types": client_df["Item Type"].value_counts().head(3).to_dict(),
                "top_item_categories": client_df["Item Category"].value_counts().head(3).to_dict(),
            }
        )

    pairwise_gaps = {}
    for left, right in combinations(range(len(client_indices)), 2):
        gap = abs(blocked_rates[left] - blocked_rates[right])
        pairwise_gaps[f"{left}-{right}"] = float(gap)

    return {
        "n_clients": len(client_indices),
        "min_client_size": int(min(stat["size"] for stat in client_stats)),
        "max_client_size": int(max(stat["size"] for stat in client_stats)),
        "blocked_rate_min": float(min(blocked_rates)),
        "blocked_rate_max": float(max(blocked_rates)),
        "pairwise_label_rate_gaps": pairwise_gaps,
        "clients": client_stats,
    }


def write_split_audit(raw_train_df, client_indices, output_path):
    audit = build_split_audit(raw_train_df, client_indices)
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    return audit


def save_default_partitions(
    processed_train_path="train_data.pkl",
    raw_train_path="raw_train_data.pkl",
    output_dir=".",
    num_clients=10,
    seed=42,
    leniency=0.1,
    min_client_size=1000,
    train_prop=0.8,
):
    dataset, _ = load_pkl_as_dataset(processed_train_path)
    raw_train_df = load_case_frame(raw_train_path)

    client_indices, keep_mask = balanced_feature_clusters(
        dataset,
        num_clients=num_clients,
        seed=seed,
        leniency=leniency,
        min_client_size=min_client_size,
    )
    client_train_idx, client_val_idx = client_train_val_split(
        client_indices,
        train_prop=train_prop,
        seed=seed,
        labels=np.asarray(dataset.y),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "partitions.pkl", "wb") as handle:
        pickle.dump(
            {
                "client_train_idx": client_train_idx,
                "client_val_idx": client_val_idx,
            },
            handle,
        )

    audit = write_split_audit(raw_train_df, client_indices, output_dir / "split_audit.json")
    return client_train_idx, client_val_idx, audit, keep_mask


def main():
    args = parse_args()
    train_df, test_df, metadata = save_processed_artifacts(
        raw_train_path=args.raw_train,
        raw_test_path=args.raw_test,
        output_dir=args.output_dir,
    )
    print(f"wrote train split with shape {train_df.shape} to {args.output_dir}")
    print(f"wrote test split with shape {test_df.shape} to {args.output_dir}")
    print(f"input size: {metadata['input_size']} to {args.output_dir}")


if __name__ == "__main__":
    main()
