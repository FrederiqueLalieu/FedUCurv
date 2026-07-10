import os
import pickle
import numpy as np
import pandas as pd
import json
import argparse

def generate_dataframe(X, y):
    num_features = X.shape[1]
    print(num_features)
    names = [f'feature{n}' for n in range(num_features)]
    print(names)
    df_dict = {names[i] : X[:, i] for i in range(num_features)}
    df_dict['was_blocked'] = y
    print(df_dict)
    train_df = pd.DataFrame(df_dict)
    return train_df

def generate_metadata(num_features):
    names = [f'feature{n}' for n in range(num_features)]
    return {
        "target_column": 'was_blocked',
        "categorical_features": [],
        "numeric_features": names,
        "ignored_columns": [],
        "feature_names": names,
        "input_size": num_features,
    }

def generate_logistic_federated_data(
    num_clients,
    client_sizes,
    client_means,
    client_stds,
    client_as,
    client_bs,
    data_dir,
    partitions_dir,
    train_fraction=0.8,
    val_fraction=0.2,
    random_seed=None,
):
    """
    Generate synthetic federated data with a logistic label model.

    For client c:
        x_c ~ Normal(mean = client_means[c], std = client_stds[c])
        z_c = client_as[c] * x_c + client_bs[c]
        p_c = sigmoid(z_c)
        y_c ~ Bernoulli(p_c)

    Saves:
      - train.pkl: pandas DataFrame with columns ["feature1", "was_blocked"]
      - test.pkl:  pandas DataFrame with columns ["feature1", "was_blocked"]
      - partitions.pkl: dict with lists of global train indices per client
    """

    if not (
        len(client_sizes) == len(client_means) == len(client_stds)
        == len(client_as) == len(client_bs) == num_clients
    ):
        raise ValueError(
            "client_sizes, client_means, client_stds, client_as, client_bs "
            "must all have length num_clients."
        )

    if not (0.0 < train_fraction <= 1.0):
        raise ValueError("train_fraction must be in (0, 1].")

    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1).")

    rng = np.random.default_rng(random_seed)

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    client_train_idx = []
    client_val_idx = []
    current_train_index = 0

    def sigmoid(z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    for c in range(num_clients):
        n_total = int(client_sizes[c])
        mean = client_means[c]
        std = client_stds[c]
        a = client_as[c]
        b = client_bs[c]

        # Sample features
        # mean = (1, 2)
        # cov = [[1, 0], [0, 1]]
        x = np.random.multivariate_normal(mean, std, (n_total))
        print(x.shape)
        # x = rng.normal(loc=mean, scale=std, size=n_total)

        # Probabilities via logistic
        z = x @ np.array(a) + b
        p = sigmoid(z)

        # Sample labels
        y = rng.binomial(n=1, p=p, size=n_total)
        # y = np.expand_dims(y, axis=1)
        print(y.shape)

        # Shuffle within client
        perm = rng.permutation(n_total)
        x = x[perm]
        y = y[perm]

        # Split into train+val vs test
        n_trainval = int(np.floor(train_fraction * n_total))
        x_trainval = x[:n_trainval]
        y_trainval = y[:n_trainval]
        x_test_client = x[n_trainval:]
        y_test_client = y[n_trainval:]

        # Split train+val into client-train vs client-val
        n_client_train = int(np.floor((1.0 - val_fraction) * n_trainval))
        x_train_client = x_trainval[:n_client_train]
        y_train_client = y_trainval[:n_client_train]
        x_val_client = x_trainval[n_client_train:]
        y_val_client = y_trainval[n_client_train:]

        # Global indices (order: client-train, then client-val)
        train_indices = list(
            range(current_train_index, current_train_index + len(x_train_client))
        )
        current_train_index += len(x_train_client)

        val_indices = list(
            range(current_train_index, current_train_index + len(x_val_client))
        )
        current_train_index += len(x_val_client)

        client_train_idx.append(train_indices)
        client_val_idx.append(val_indices)

        # Append to global containers
        train_features.append(x_train_client)
        train_features.append(x_val_client)
        train_labels.append(y_train_client)
        train_labels.append(y_val_client)

        test_features.append(x_test_client)
        test_labels.append(y_test_client)

    # Concatenate global arrays
    if train_features:
        X_train = np.concatenate(train_features, axis=0)
        y_train = np.concatenate(train_labels, axis=0).astype(np.int64)
    else:
        X_train = np.empty((0,))
        y_train = np.empty((0,), dtype=np.int64)

    if test_features:
        X_test = np.concatenate(test_features, axis=0)
        y_test = np.concatenate(test_labels, axis=0).astype(np.int64)
    else:
        X_test = np.empty((0,))
        y_test = np.empty((0,), dtype=np.int64)

    # Create DataFrames with named columns
    train_df = generate_dataframe(X_train, y_train)
    test_df = generate_dataframe(X_test, y_test)
    # train_df = pd.DataFrame({
    #     "feature1": X_train,
    #     "was_blocked": y_train,
    # })

    # test_df = pd.DataFrame({
    #     "feature1": X_test,
    #     "was_blocked": y_test,
    # })

    num_features = X_train.shape[1]
    metadata = generate_metadata(num_features)


    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(partitions_dir, exist_ok=True)

    # Save data as pickled DataFrames
    train_path = os.path.join(data_dir, "train_data.pkl")
    test_path = os.path.join(data_dir, "test_data.pkl")
    partitions_path = os.path.join(partitions_dir, "partitions.pkl")
    metadata_path = os.path.join(partitions_dir, "feature_metadata.json")

    with open(train_path, "wb") as f:
        pickle.dump(train_df, f)

    with open(test_path, "wb") as f:
        pickle.dump(test_df, f)

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    with open(partitions_path, "wb") as f:
        pickle.dump(
            {
                "client_train_idx": client_train_idx,
                "client_val_idx": client_val_idx,
            },
            f,
        )

    print(f"Saved train data to: {train_path}")
    print(f"Saved test data to: {test_path}")
    print(f"Saved partitions to: {partitions_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".") 
    parser.add_argument("--experiment", default="1")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiment == '1':
        # Experiment 1
        num_clients = 3
        client_sizes = [82911, 82911, 82911]
        client_means = [[-2],[0],[2]]
        cov = [[0.5]]
        client_covs = [cov, cov, cov]
        a = [1.5]
        b = 0.0  
        client_as = [a] * num_clients
        client_bs = [b] * num_clients


    if args.experiment == '2':    
        # Experiment 2
        num_clients = 3
        client_sizes = [41956, 82911, 125866]
        client_means = [[-2],[0],[2]]
        cov = [[0.5]]
        client_covs = [cov, cov, cov]
        a = [1.5]
        b = 0.0  
        client_as = [a] * num_clients
        client_bs = [b] * num_clients

    if args.experiment =='3':
        # Experiment 3
        num_clients = 3
        client_sizes = [41956, 82911, 125866]
        cov = [[0.5, 0], [0, 0.5]]
        client_means = [[-2, -2], [0, 0],  [2, 2]]
        client_covs = [cov, cov, cov]
        a = [1.5, 1.5]
        b = 0.0  
        client_as = [a] * num_clients
        client_bs = [b] * num_clients

    if args.experiment =='4':

        num_clients = 5
        client_sizes = [50147] * num_clients
        cov = [[0.5]]
        client_means = [[-2], [-1], [0], [1], [2]]
        client_covs = [cov] * num_clients
        a = [1.5]
        b = 0.0  
        client_as = [a] * num_clients
        client_bs = [b] * num_clients

    if args.experiment =='5':

        num_clients = 10
        client_sizes = [25073] * num_clients
        cov = [[0.5]]
        client_means = [[-5], [4], [-4], [3], [-3], [-2], [-1], [0], [1], [2]]
        client_covs = [cov] * num_clients
        a = [1.5]
        b = 0.0  
        client_as = [a] * num_clients
        client_bs = [b] * num_clients


    generate_logistic_federated_data(
        num_clients=num_clients,
        client_sizes=client_sizes,
        client_means=client_means,
        client_stds=client_covs,
        client_as=client_as,
        client_bs=client_bs,
        data_dir= args.dir,
        partitions_dir= args.dir,
        train_fraction=0.8,
        val_fraction=0.2,
        random_seed=42)

if __name__ == '__main__':
    main()