import torch
from copy import deepcopy

from data_utils import *
from training import *


def fed_avg(state_dicts, data_sizes):
    """
    Takes the local model parameters and outputs the average for each parameter.
    """
    total_size = sum(data_sizes)
    avg_state = {}
    for key in state_dicts[0]:
        avg_state[key] = sum(
            sd[key] * (size / total_size)
            for sd, size in zip(state_dicts, data_sizes)
        )
    return avg_state

def state_dict_distance(state1, state2):
    diff_vecs = []
    base_vecs = []

    for k in state1.keys():
        t1 = state1[k]
        t2 = state2[k]

        d = (t2 - t1).float().view(-1)
        diff_vecs.append(d)
        base_vecs.append(t1.float().view(-1))

    diff_vec = torch.cat(diff_vecs)
    base_vec = torch.cat(base_vecs)

    dist = diff_vec.norm(2)

    return dist

def simulate_federated_learning(
    client_train_loaders,
    client_val_loaders,
    device,
    epochs=2,
    lr=0.01,
    optimizer_name="sgd",
    weight_decay=0.0,
    momentum=0.0,
    pos_weight=None,
    batch_size=32,
    alpha=0.1,
    max_rounds=50,
    loss_thresh=0.01,
    acc_thresh=0.01,
    min_rounds=6,
    patience=3,
    pretrained=None,
    init_seed=None,
    run_label=None,
    progress=False,
):
    """
    Simulates a specified federated learning, until convergence with a maximum
    number of rounds. Training is based on a user specified data set, which is
    divided into clients within the function.
    """

    input_size = client_train_loaders[0].dataset[0][0].shape[-1]
    data_sizes = [len(dataloader.dataset) for dataloader in client_train_loaders]
    data_size = sum(data_sizes)
    num_clients = len(client_train_loaders)

    # Initialize global model
    if init_seed is not None:
        torch.manual_seed(int(init_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(init_seed))

    global_model = Net(input_size=input_size).to(device)
    if pretrained is not None:
        global_model.load_state_dict(pretrained)
    global_losses = []
    global_accuracies = []

    # Start tracking for unlearning
    state_dict = global_model.state_dict()
    summaries = [{k: torch.zeros_like(v) for k, v in state_dict.items()} for _ in range(num_clients)]
    curvature = {k: torch.zeros_like(v) for k, v in state_dict.items()}

    converged = False
    stale_rounds = 0
    best_loss = float("inf")
    best_acc = float("-inf")
    best_state = deepcopy(global_model.state_dict())
    best_results = None

    delta_trajectory = [] # for each round, for each parameter, the average absolute delta of all the clients
    summary_trajectory = [] # for each round, for each parameter, the average of all the clients
    convergence_metrics = [] # for each round, the difference between gloabal model this round and previous round
    client_delta_rounds = [] # for each round, for each client, for each parameter, the delta
    
    if progress and run_label is not None:
        print(f"[{run_label}] start federated run with {max_rounds} rounds", flush=True)

    for round_idx in range(max_rounds):

        global_state = global_model.state_dict()
        client_states = []
        client_deltas = []
        scaled_deltas = []

        # Local training
        for i in range(num_clients):
            local_model = Net(input_size=input_size).to(device)
            local_model.load_state_dict(global_state)

            train_one_model(
                local_model,
                client_train_loaders[i],
                device,
                epochs,
                lr,
                optimizer_name=optimizer_name,
                weight_decay=weight_decay,
                momentum=momentum,
                pos_weight=pos_weight,
            )
            client_states.append(local_model.state_dict())

            delta = {k: global_state[k] - client_states[i][k] for k in global_state}
            scaled_delta = {k: (data_sizes[i] / data_size) * delta[k] for k in delta}
            client_deltas.append(delta)
            scaled_deltas.append(scaled_delta)

        avg_delta = {
            k: torch.sum(torch.stack([d[k] for d in scaled_deltas]), dim=0)
            for k in scaled_deltas[0]
        }

        for i in range(num_clients):
            summaries[i] = {
                k: summaries[i][k] + client_deltas[i][k]
                for k in client_deltas[i]
            }

        client_delta_rounds.append(
            [
                {k: value.detach().clone() for k, value in client_deltas[i].items()}
                for i in range(num_clients)
            ]
        )

        avg_summary = {
            k: float(torch.stack([summaries[i][k] for i in range(num_clients)]).abs().mean().item())
            for k in avg_delta
        }
        round_delta = {k: float(avg_delta[k].abs().mean().item()) for k in avg_delta}

        summary_trajectory.append(avg_summary)
        delta_trajectory.append(round_delta)

        curvature = {
            k: (1 - alpha) * curvature[k]
            + alpha
            * (1 / batch_size)
            * torch.sum(
                torch.stack(
                    [
                        (data_sizes[i] / data_size) * (client_deltas[i][k] ** 2)
                        for i in range(len(client_deltas))
                    ]
                ),
                dim=0,
            )
            for k in avg_delta
        }

        # Federated averaging
        new_global_state = fed_avg(client_states, data_sizes)
        state_diff = state_dict_distance(new_global_state, global_state)
        # state_diff = {k: new_global_state[k] - global_state[k] for k in global_state}

        global_model.load_state_dict(new_global_state)

        # state_diff = {k: new_global_state[k] - global_state[k] for k in global_state}
        convergence_metrics.append(state_diff)

        # Validation

        results = create_val_results_dict(global_model, client_val_loaders, device)

        len_valloaders = [len(valloader.dataset) for valloader in client_val_loaders]
        losses = results['loss']
        accuracies = results['acc']

        total_val_loss = sum([loss * n for loss, n in zip(losses, len_valloaders)])
        total_correct = sum([acc * n for acc, n in zip(accuracies, len_valloaders)])
        total_samples = sum(len_valloaders)

        avg_loss = total_val_loss / total_samples
        avg_acc = total_correct / total_samples

        global_losses.append(avg_loss)
        global_accuracies.append(avg_acc)

        improved_loss = avg_loss < (best_loss - loss_thresh)
        improved_acc = avg_acc > (best_acc + acc_thresh)
        if improved_loss or improved_acc or best_results is None:
            if improved_loss or (
                abs(avg_loss - best_loss) <= loss_thresh and avg_acc > best_acc
            ) or best_results is None:
                best_state = deepcopy(global_model.state_dict())
                best_results = results
                best_loss = avg_loss
                best_acc = avg_acc
            else:
                best_acc = max(best_acc, avg_acc)
            stale_rounds = 0
        else:
            stale_rounds += 1

        if (round_idx + 1) >= min_rounds and stale_rounds >= patience:
            if run_label is not None:
                print(f"[{run_label}] early stop after {round_idx + 1} rounds.", flush=True)
            converged = True

        if progress and run_label is not None:
            print(
                f"[{run_label}] round {round_idx + 1}/{max_rounds} "
                f"val_loss={avg_loss:.4f} val_acc={avg_acc:.4f}",
                flush=True,
            )

        if converged:
            break

    if best_results is not None:
        global_model.load_state_dict(best_state)
        results = best_results

    if converged is False:
        if run_label is not None:
            print(f"[{run_label}] didn't converge within {max_rounds} rounds.", flush=True)
        else:
            print(f"Didn't converge within {max_rounds} rounds.")

    return (
        global_model,
        summaries,
        curvature,
        results,
        delta_trajectory,
        summary_trajectory,
        convergence_metrics,
        client_delta_rounds,
    )
