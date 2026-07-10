import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class Net(nn.Module):
    def __init__(self, input_size=14):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 5)
        self.fc4 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def safe_binary_auc(labels, probs):
    labels = labels.reshape(-1)
    probs = probs.reshape(-1)
    finite_mask = torch.isfinite(labels) & torch.isfinite(probs)
    labels = labels[finite_mask]
    probs = probs[finite_mask]

    if labels.numel() == 0 or torch.unique(labels).numel() < 2:
        return float("nan")

    return float(roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy()))


def expected_calibration_error(probs, labels, n_bins=10):
    probs = probs.reshape(-1)
    labels = labels.reshape(-1)
    finite_mask = torch.isfinite(probs) & torch.isfinite(labels)
    probs = probs[finite_mask]
    labels = labels[finite_mask]

    if probs.numel() == 0:
        return float("nan")

    bin_edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)

    for idx in range(n_bins):
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]

        if idx == n_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)

        if not torch.any(mask):
            continue

        weight = mask.float().mean()
        avg_conf = probs[mask].mean()
        avg_target = labels[mask].mean()
        ece += weight * torch.abs(avg_conf - avg_target)

    return float(ece.item())


def make_loss(device, pos_weight=None):
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()

    weight = torch.tensor(float(pos_weight), dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=weight)


def make_optimizer(model, lr, optimizer_name="sgd", weight_decay=0.0, momentum=0.0):
    name = optimizer_name.lower()

    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )

    if name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(weight_decay),
        )

    raise ValueError(f"unknown optimizer: {optimizer_name}")


def train_one_model(
    model,
    train_loader,
    device,
    epochs,
    lr,
    optimizer_name="sgd",
    weight_decay=0.0,
    momentum=0.0,
    pos_weight=None,
):

    criterion = make_loss(device, pos_weight=pos_weight)
    optimizer = make_optimizer(
        model,
        lr,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    model.to(device)
    model.train()
    for _ in range(epochs):
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device).float()
            optimizer.zero_grad()
            logits = model(X).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    correct, loss_sum, tp, pred_pos, actual_pos = 0, 0.0, 0, 0, 0
    with torch.no_grad():
        for X, y in testloader:
            X = X.to(device)
            y = y.to(device).float()
            outputs = net(X).squeeze(-1)
            loss_sum += float(criterion(outputs, y).item()) * len(y)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += int((preds == y).sum().item())

            predicted_positive = preds == 1
            actual_positive = y == 1
            tp += int((predicted_positive & actual_positive).sum().item())
            pred_pos += int(predicted_positive.sum().item())
            actual_pos += int(actual_positive.sum().item())

    accuracy = correct / len(testloader.dataset)
    precision = float(tp / (pred_pos + 1e-8))
    recall = float(tp / (actual_pos + 1e-8))
    loss = loss_sum / len(testloader.dataset)
    return loss, accuracy, precision, recall

def evaluate(model, data_loader, device):

    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    model.to(device)
    dataset_size = len(data_loader.dataset)
    correct, total_loss = 0, 0.0
    tp, pred_pos, actual_pos = 0, 0, 0
    confidence_sum, brier_sum = 0.0, 0.0
    probs_all = []
    labels_all = []
    valid_count = 0
    non_finite_count = 0

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device).float()

            logits = model(X).squeeze(-1)
            probs = torch.sigmoid(logits)
            finite_mask = torch.isfinite(logits) & torch.isfinite(probs) & torch.isfinite(y)
            non_finite_count += int((~finite_mask).sum().item())
            actual_pos += int((y == 1).sum().item())

            if not torch.any(finite_mask):
                continue

            logits = logits[finite_mask]
            probs = probs[finite_mask]
            y = y[finite_mask]
            loss = criterion(logits, y)
            preds = (probs > 0.5).float()

            total_loss += loss.item() * len(y)
            valid_count += len(y)
            correct += (preds == y).float().sum().item()

            predicted_positive = preds == 1
            actual_positive = y == 1
            tp += int((predicted_positive & actual_positive).sum().item())
            pred_pos += int(predicted_positive.sum().item())

            confidence_sum += float(torch.maximum(probs, 1 - probs).sum().item())
            brier_sum += float(((probs - y) ** 2).sum().item())
            probs_all.append(probs.cpu())
            labels_all.append(y.cpu())

    if probs_all:
        probs_all = torch.cat(probs_all)
        labels_all = torch.cat(labels_all)
    else:
        probs_all = torch.tensor([])
        labels_all = torch.tensor([])

    avg_loss = float("nan") if valid_count == 0 else total_loss / valid_count
    accuracy = correct / dataset_size
    precision = float(tp / (pred_pos + 1e-8))
    recall = float(tp / (actual_pos + 1e-8))
    f1 = float((2 * precision * recall) / (precision + recall + 1e-8))
    avg_confidence = float("nan") if valid_count == 0 else confidence_sum / valid_count
    brier = float("nan") if valid_count == 0 else brier_sum / valid_count
    positive_rate = float(pred_pos / dataset_size)
    auc = safe_binary_auc(labels_all, probs_all)
    ece = expected_calibration_error(probs_all, labels_all)
    valid_rate = valid_count / dataset_size
    non_finite_rate = non_finite_count / dataset_size

    return avg_loss, accuracy, precision, recall, avg_confidence, f1, auc, brier, ece, positive_rate, valid_rate, non_finite_rate

from data_utils import (
    load_pkl_as_dataset,
    global_train_val_split,
    make_dataloaders,
)

def train_global(train_path,
                 batch_size=32,
                 epochs=10,
                 lr=0.01,
                 train_prop=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, metadata = load_pkl_as_dataset(train_path)

    train_ds, val_ds, train_idx, val_idx = global_train_val_split(
        dataset, train_prop=train_prop
    )
    train_loader, val_loader = make_dataloaders(train_ds, val_ds, batch_size)

    input_size = metadata["input_size"]
    model = Net(input_size=input_size)

    train_one_model(model, train_loader, device, epochs, lr)

    val_loss, val_acc, *_ = evaluate(model, val_loader, device)
    print(f"Global validation - loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    return model, metadata, train_idx, val_idx

def create_val_results_dict(model, val_loaders, device):

    loss_per_client = []
    accuracy_per_client = []
    prec_per_client = []
    recall_per_client = []
    confidence_per_client = []
    f1_per_client = []
    auc_per_client = []
    brier_per_client = []
    ece_per_client = []
    positive_rate_per_client = []
    valid_rate_per_client = []
    non_finite_rate_per_client = []

    model.eval()

    for val_loader in val_loaders:
        loss, acc, prec, recall, conf, f1, auc, brier, ece, pos_rate, valid_rate, non_finite_rate = evaluate(model, val_loader, device)
        loss_per_client.append(loss)
        accuracy_per_client.append(acc)
        prec_per_client.append(prec)
        recall_per_client.append(recall)
        confidence_per_client.append(conf)
        f1_per_client.append(f1)
        auc_per_client.append(auc)
        brier_per_client.append(brier)
        ece_per_client.append(ece)
        positive_rate_per_client.append(pos_rate)
        valid_rate_per_client.append(valid_rate)
        non_finite_rate_per_client.append(non_finite_rate)


    results = {'loss': loss_per_client,
               'acc': accuracy_per_client,
               'recall': recall_per_client,
               'prec': prec_per_client,
               'confidence': confidence_per_client,
               'f1': f1_per_client,
               'auc': auc_per_client,
               'brier': brier_per_client,
               'ece': ece_per_client,
               'positive_rate': positive_rate_per_client,
               'valid_rate': valid_rate_per_client,
               'non_finite_rate': non_finite_rate_per_client}
    
    return results

def create_test_results_dict(model, test_loader, device):

    loss, acc, prec, recall, conf, f1, auc, brier, ece, pos_rate, valid_rate, non_finite_rate = evaluate(model, test_loader, device)
    results = {'loss': loss,
               'acc': acc,
               'recall': recall,
               'prec': prec,
               'confidence': conf,
               'f1': f1,
               'auc': auc,
               'brier': brier,
               'ece': ece,
               'positive_rate': pos_rate,
               'valid_rate': valid_rate,
               'non_finite_rate': non_finite_rate}
    
    return results
