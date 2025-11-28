#!/usr/bin/env python3
"""
Train Traditional model (fixed-order residual layers).

This is a standard transformer-style model where layers are always
applied in the same order: L0 → L1 → ... → L_{n-1}.

Usage:
    uv run python -m cogitatio.scripts.train_traditional --epochs 20 --n_layers 4
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cogitatio.dataset import create_dataloaders, VOCAB_SIZE
from cogitatio.model import TraditionalModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Traditional (fixed-order layers) model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model architecture
    parser.add_argument("--d_model", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of residual layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--pool_mode", type=str, default="mean", choices=["mean", "last"],
                        help="Pooling mode before decoder (mean or last token)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    
    # Data
    parser.add_argument("--train_samples", type=int, default=100_000, help="Training samples")
    parser.add_argument("--val_samples", type=int, default=10_000, help="Validation samples")
    parser.add_argument("--test_samples", type=int, default=10_000, help="Test samples")
    parser.add_argument("--w_local", type=float, default=1.0, help="Weight for LOCAL task")
    parser.add_argument("--w_global", type=float, default=1.0, help="Weight for GLOBAL task")
    parser.add_argument("--w_second_min", type=float, default=1.0, help="Weight for SECOND_MIN task")
    parser.add_argument("--global_weight", type=float, default=1.0, 
                        help="Loss weight for GLOBAL samples (>1 to prioritize harder task)")
    
    # Device
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output
    parser.add_argument("--save_model", type=str, default=None, help="Path to save model")
    
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def train_epoch(model, loader, optimizer, device, grad_clip, global_weight=1.0):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        modes = batch["mode"].to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        
        # Per-sample loss with task weighting
        per_sample_loss = F.cross_entropy(logits, y, reduction='none')
        
        # Weight GLOBAL samples higher to balance task difficulty
        weights = torch.ones_like(per_sample_loss)
        weights[modes == 1] = global_weight  # GLOBAL = mode 1
        
        loss = (per_sample_loss * weights).mean()
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += per_sample_loss.mean().item() * x.shape[0]  # Unweighted for logging
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += x.shape[0]
    
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Per-mode metrics
    local_correct, local_total = 0, 0
    global_correct, global_total = 0, 0
    second_min_correct, second_min_total = 0, 0
    
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        modes = batch["mode"].to(device)
        
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        correct = (preds == y)
        
        total_loss += loss.item() * x.shape[0]
        total_correct += correct.sum().item()
        total_samples += x.shape[0]
        
        # Per-mode
        local_mask = (modes == 0)
        global_mask = (modes == 1)
        second_min_mask = (modes == 2)
        local_correct += correct[local_mask].sum().item()
        local_total += local_mask.sum().item()
        global_correct += correct[global_mask].sum().item()
        global_total += global_mask.sum().item()
        second_min_correct += correct[second_min_mask].sum().item()
        second_min_total += second_min_mask.sum().item()
    
    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
        "acc_local": local_correct / max(local_total, 1),
        "acc_global": global_correct / max(global_total, 1),
        "acc_second_min": second_min_correct / max(second_min_total, 1),
        "n_second_min": second_min_total,
    }


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = get_device(args.device)
    
    print("=" * 60)
    print("Traditional Model Training (Fixed-Order Layers)")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Configuration: {vars(args)}\n")
    
    # Data
    mode_weights = (args.w_local, args.w_global, args.w_second_min)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        mode_weights=mode_weights,
    )
    
    # Model
    model = TraditionalModel(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        pool_mode=args.pool_mode,
    ).to(device)
    
    print(f"Parameters: {model.count_parameters()}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, args.grad_clip, args.global_weight
        )
        val_metrics = evaluate(model, val_loader, device)
        
        acc_str = f"(L: {val_metrics['acc_local']:.4f}, G: {val_metrics['acc_global']:.4f}"
        if val_metrics.get('n_second_min', 0) > 0:
            acc_str += f", S: {val_metrics['acc_second_min']:.4f}"
        acc_str += ")"
        print(f"Epoch {epoch:2d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f} "
              f"{acc_str}")
        
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
    
    # Test
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    test_metrics = evaluate(model, test_loader, device)
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['acc']:.4f}")
    print(f"  Local:     {test_metrics['acc_local']:.4f}")
    print(f"  Global:    {test_metrics['acc_global']:.4f}")
    if test_metrics.get('n_second_min', 0) > 0:
        print(f"  SecondMin: {test_metrics['acc_second_min']:.4f}")
    
    if args.save_model:
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "test_metrics": test_metrics,
        }, args.save_model)
        print(f"\nModel saved to {args.save_model}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

