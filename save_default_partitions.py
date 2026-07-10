import argparse
from data_utils import save_default_partitions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train_data.pkl")
    parser.add_argument("--raw-train", default="raw_train_data.pkl")
    parser.add_argument("--num-clients", default=10)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--leniency", default=0.1)
    parser.add_argument("--min-size", default=1000)
    parser.add_argument("--train-prop", default=0.8)
    parser.add_argument("--output-dir", default='.')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    client_train_idx, client_val_idx, split_audit, kept_features = save_default_partitions(
        processed_train_path=args.train,
        raw_train_path=args.raw_train,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        seed=args.seed,
        leniency=args.leniency,
        min_client_size=args.min_size,
        train_prop=args.train_prop,
    )