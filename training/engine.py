import torch
from tqdm import tqdm

from evaluation.metrics import binary_metrics


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    model.train()
    total_loss = 0.0
    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images).view(-1)
            loss = loss_fn(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / max(1, len(loader.dataset))


def train_npr_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    """Train one NPR epoch and reject non-finite logits/losses."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    for images, labels, _ in tqdm(loader, desc="train-npr", leave=False):
        images = images.to(device)
        labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images).view(-1)
            loss = loss_fn(logits, labels)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError("NPR training produced NaN or Inf")
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)
    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate_loader(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    labels_all, probs_all, paths_all = [], [], []
    for images, labels, paths in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        labels_t = torch.as_tensor(labels, dtype=torch.float32, device=device)
        logits = model(images).view(-1)
        loss = loss_fn(logits, labels_t)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        total_loss += loss.item() * labels_t.size(0)
        labels_all.extend(labels_t.cpu().numpy().astype(int).tolist())
        probs_all.extend(probs.tolist())
        paths_all.extend(paths)
    metrics = binary_metrics(labels_all, probs_all)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    metrics["labels"] = labels_all
    metrics["probs"] = probs_all
    metrics["paths"] = paths_all
    return metrics


def train_feature_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for features, labels in tqdm(loader, desc="train-mlp", leave=False):
        features = features.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features).view(-1)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate_feature_loader(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    labels_all, probs_all = [], []
    for features, labels in tqdm(loader, desc="eval-mlp", leave=False):
        features = features.to(device)
        labels = labels.to(device).float()
        logits = model(features).view(-1)
        loss = loss_fn(logits, labels)
        probs = torch.sigmoid(logits).cpu().numpy()
        total_loss += loss.item() * labels.size(0)
        labels_all.extend(labels.cpu().numpy().astype(int).tolist())
        probs_all.extend(probs.tolist())
    metrics = binary_metrics(labels_all, probs_all)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    metrics["labels"] = labels_all
    metrics["probs"] = probs_all
    return metrics
