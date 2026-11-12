import torch

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for patches, masks in dataloader:
        patches, masks = patches.to(device), masks.to(device)
        
        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, masks)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * patches.size(0)
        
        # Fast GPU tensor-based dice without multi-item CPU sync
        with torch.no_grad():
            preds = (logits > 0.0).float()
            inter = (preds * masks).sum()
            union = preds.sum() + masks.sum() + 1e-8
            running_dice += ((2.0 * inter) / union).item() * patches.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)
    return epoch_loss, epoch_dice

def validate_model(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    tp_tensor = torch.zeros(1, device=device)
    fp_tensor = torch.zeros(1, device=device)
    fn_tensor = torch.zeros(1, device=device)

    with torch.no_grad():
        for patches, masks in dataloader:
            patches, masks = patches.to(device), masks.to(device)
            logits = model(patches)
            loss = criterion(logits, masks)

            running_loss += loss.item() * patches.size(0)
            preds = (torch.sigmoid(logits) >= threshold).float()
            tp_tensor += (preds * masks).sum()
            fp_tensor += (preds * (1.0 - masks)).sum()
            fn_tensor += ((1.0 - preds) * masks).sum()

    eps = 1e-8
    tp = tp_tensor.item()
    fp = fp_tensor.item()
    fn = fn_tensor.item()

    val_loss = running_loss / len(dataloader.dataset)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return val_loss, {"dice": dice, "iou": iou, "precision": precision, "recall": recall}
