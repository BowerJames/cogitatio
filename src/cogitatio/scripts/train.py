#!/usr/bin/env python3
"""
Train MuThought on the toy digit sequence task.

This script trains a MuThought model using actor-critic RL to learn
adaptive computation on a synthetic task where:
- MODE_LOCAL (easy): target = last digit
- MODE_GLOBAL (medium): target = max(digits)
- MODE_SECOND_MIN (hard): target = second smallest digit

The hypothesis is that the model should learn to spend more thinking
steps on harder tasks.

Usage:
    python -m cogitatio.scripts.train --epochs 20 --n_blocks 4
    
    # Or with uv:
    uv run python -m cogitatio.scripts.train --epochs 20
    
    # With custom mode weights (default is uniform):
    uv run python -m cogitatio.scripts.train --w_local 1 --w_global 2 --w_second_min 1
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from cogitatio.dataset import create_dataloaders, VOCAB_SIZE
from cogitatio.model import MuThought
from cogitatio.trainer import MuThoughtTrainer, TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MuThought on toy digit sequence task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model architecture
    parser.add_argument(
        "--d_model", type=int, default=64,
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--n_blocks", type=int, default=4,
        help="Number of thought blocks in the pool"
    )
    parser.add_argument(
        "--n_heads", type=int, default=4,
        help="Number of attention heads per block"
    )
    parser.add_argument(
        "--max_steps", type=int, default=6,
        help="Maximum thinking steps allowed"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate"
    )
    parser.add_argument(
        "--pool_mode", type=str, default="mean",
        choices=["mean", "last"],
        help="Pooling mode before decoder (mean or last token)"
    )
    
    # Training
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr_model", type=float, default=1e-3,
        help="Learning rate for model (embeddings, blocks, decoder)"
    )
    parser.add_argument(
        "--lr_policy", type=float, default=3e-3,
        help="Learning rate for policy/value heads"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0,
        help="Gradient clipping threshold"
    )
    
    # RL hyperparameters
    parser.add_argument(
        "--step_penalty", type=float, default=0.01,
        help="Penalty per thinking step (lambda)"
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01,
        help="Entropy bonus coefficient for exploration"
    )
    parser.add_argument(
        "--value_coef", type=float, default=0.5,
        help="Value loss coefficient"
    )
    
    # Data
    parser.add_argument(
        "--train_samples", type=int, default=100_000,
        help="Number of training samples"
    )
    parser.add_argument(
        "--val_samples", type=int, default=10_000,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--test_samples", type=int, default=10_000,
        help="Number of test samples"
    )
    parser.add_argument(
        "--w_local", type=float, default=1.0,
        help="Weight for LOCAL (easy) samples"
    )
    parser.add_argument(
        "--w_global", type=float, default=1.0,
        help="Weight for GLOBAL (medium) samples"
    )
    parser.add_argument(
        "--w_second_min", type=float, default=1.0,
        help="Weight for SECOND_MIN (hard) samples"
    )
    
    # Device
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to train on"
    )
    
    # Output
    parser.add_argument(
        "--save_model", type=str, default=None,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("=" * 60)
    print("MuThought Level 2 Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()
    
    # Create dataloaders
    print("Creating datasets...")
    mode_weights = (args.w_local, args.w_global, args.w_second_min)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        mode_weights=mode_weights,
    )
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    print()
    
    # Create model
    print("Creating model...")
    model = MuThought(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        max_steps=args.max_steps,
        dropout=args.dropout,
        pool_mode=args.pool_mode,
    )
    
    param_counts = model.count_parameters()
    print(f"  Parameters: {param_counts['total']:,}")
    for name, count in param_counts.items():
        if name != "total":
            print(f"    {name}: {count:,}")
    print()
    
    # Create trainer
    config = TrainConfig(
        lr_model=args.lr_model,
        lr_policy=args.lr_policy,
        step_penalty=args.step_penalty,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        device=args.device,
    )
    
    trainer = MuThoughtTrainer(model, config)
    
    # Train
    print("=" * 60)
    print("Training")
    print("=" * 60)
    history = trainer.train(train_loader, val_loader)
    
    # Final evaluation on test set
    print("=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"\nTest Results:")
    print(f"  Overall Accuracy: {test_metrics.acc_final:.4f} (base: {test_metrics.acc_base:.4f})")
    print(f"  Local Accuracy:   {test_metrics.acc_local_final:.4f} (base: {test_metrics.acc_local_base:.4f})")
    print(f"  Global Accuracy:  {test_metrics.acc_global_final:.4f} (base: {test_metrics.acc_global_base:.4f})")
    if test_metrics.n_second_min > 0:
        print(f"  SecondMin Acc:    {test_metrics.acc_second_min_final:.4f} (base: {test_metrics.acc_second_min_base:.4f})")
    print(f"  Average Steps:    {test_metrics.avg_steps:.2f}")
    print(f"    Local:     {test_metrics.avg_steps_local:.2f}")
    print(f"    Global:    {test_metrics.avg_steps_global:.2f}")
    if test_metrics.n_second_min > 0:
        print(f"    SecondMin: {test_metrics.avg_steps_second_min:.2f}")
    print(f"  Cross-Entropy:    {test_metrics.ce_loss_final:.4f} (base: {test_metrics.ce_loss_base:.4f})")
    
    # Adaptive compute check
    print(f"\nAdaptive Compute Analysis:")
    if test_metrics.avg_steps_global > test_metrics.avg_steps_local:
        diff = test_metrics.avg_steps_global - test_metrics.avg_steps_local
        print(f"  Model uses {diff:.2f} more steps on GLOBAL vs LOCAL tasks (good!)")
    else:
        print(f"  Model does NOT use more steps on GLOBAL tasks (may need tuning)")
    
    if test_metrics.n_second_min > 0:
        if test_metrics.avg_steps_second_min > test_metrics.avg_steps_global:
            diff = test_metrics.avg_steps_second_min - test_metrics.avg_steps_global
            print(f"  Model uses {diff:.2f} more steps on SECOND_MIN vs GLOBAL tasks (good!)")
        else:
            print(f"  Model does NOT use more steps on SECOND_MIN tasks (may need tuning)")
    
    # Block usage
    print(f"\nBlock Usage:")
    for i in range(model.n_blocks):
        usage = test_metrics.block_usage.get(i, 0)
        print(f"  Block {i}: {usage:.1%}")
    print(f"  STOP: {test_metrics.block_usage.get(model.stop_action, 0):.1%}")
    
    # Save model
    if args.save_model:
        save_path = Path(args.save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "test_metrics": {
                "acc_final": test_metrics.acc_final,
                "acc_base": test_metrics.acc_base,
                "acc_local_final": test_metrics.acc_local_final,
                "acc_global_final": test_metrics.acc_global_final,
                "acc_second_min_final": test_metrics.acc_second_min_final,
                "avg_steps": test_metrics.avg_steps,
                "avg_steps_local": test_metrics.avg_steps_local,
                "avg_steps_global": test_metrics.avg_steps_global,
                "avg_steps_second_min": test_metrics.avg_steps_second_min,
            },
        }, save_path)
        print(f"\nModel saved to {save_path}")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

