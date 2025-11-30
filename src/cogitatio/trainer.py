"""
MuThought Trainer with Actor-Critic training.

Training loop that:
1. Computes zero-thought baseline loss
2. Runs thinking loop to get final prediction
3. Computes reward = improvement over baseline - step penalty
4. Updates policy with REINFORCE + value baseline
5. Updates value head with MSE loss
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MuThought, ThoughtOutput


@dataclass
class TrainConfig:
    """Training configuration."""
    # Learning rates
    lr_model: float = 1e-3      # LR for embeddings, layers, decoder
    lr_policy: float = 3e-3     # LR for policy/value heads (faster)
    
    # RL hyperparameters
    step_penalty: float = 0.01  # Cost per thinking step
    entropy_coef: float = 0.01  # Entropy bonus for exploration
    value_coef: float = 0.5     # Value loss coefficient
    
    # Training
    epochs: int = 20
    grad_clip: float = 1.0
    
    # Device
    device: str = "auto"


@dataclass
class Metrics:
    """Training metrics container."""
    # Losses
    total_loss: float = 0.0
    ce_loss_base: float = 0.0
    ce_loss_final: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    
    # Accuracy
    acc_base: float = 0.0
    acc_final: float = 0.0
    acc_local_base: float = 0.0
    acc_local_final: float = 0.0
    acc_global_base: float = 0.0
    acc_global_final: float = 0.0
    acc_second_min_base: float = 0.0
    acc_second_min_final: float = 0.0
    
    # Steps
    avg_steps: float = 0.0
    avg_steps_local: float = 0.0
    avg_steps_global: float = 0.0
    avg_steps_second_min: float = 0.0
    
    # Layer usage (will be populated)
    layer_usage: Dict[int, float] = field(default_factory=dict)
    
    # Counts for averaging
    n_samples: int = 0
    n_local: int = 0
    n_global: int = 0
    n_second_min: int = 0


class MuThoughtTrainer:
    """
    Trainer for MuThought with actor-critic learning.
    
    The training objective:
    - Reward R = L_base - L_final - lambda * num_steps
      (positive reward means thinking helped)
    - Policy loss: -log_prob * advantage (REINFORCE)
    - Value loss: MSE(predicted_value, actual_return)
    - Also minimize CE loss on final predictions
    """
    
    def __init__(
        self,
        model: MuThought,
        config: TrainConfig,
    ):
        self.model = model
        self.config = config
        
        # Setup device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model = self.model.to(self.device)
        
        # Create parameter groups with different learning rates
        # Slow: embeddings, latent layers, decoder, norm
        slow_params = list(model.token_emb.parameters()) + \
                      list(model.pos_emb.parameters()) + \
                      list(model.step_emb.parameters()) + \
                      list(model.block_emb.parameters()) + \
                      list(model.latent_layers.parameters()) + \
                      list(model.decoder.parameters()) + \
                      list(model.norm.parameters())
        
        # Fast: policy and value heads
        fast_params = list(model.policy_head.parameters()) + \
                      list(model.value_head.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {"params": slow_params, "lr": config.lr_model},
            {"params": fast_params, "lr": config.lr_policy},
        ])
        
        # Learning rate scheduler
        self.scheduler = None
    
    def compute_loss(
        self,
        output: ThoughtOutput,
        targets: torch.Tensor,
        modes: torch.Tensor,
    ) -> tuple:
        """
        Compute all loss components.
        
        Args:
            output: ThoughtOutput from model forward
            targets: (batch,) ground truth targets
            modes: (batch,) 0=local, 1=global
            
        Returns:
            (total_loss, metrics_dict)
        """
        batch_size = targets.shape[0]
        
        # Cross-entropy losses
        ce_loss_base = F.cross_entropy(output.logits_base, targets, reduction="none")
        ce_loss_final = F.cross_entropy(output.logits_final, targets, reduction="none")
        
        # Reward: improvement minus step penalty
        # R = L_base - L_final - lambda * steps
        # Positive means thinking helped
        reward = ce_loss_base - ce_loss_final - self.config.step_penalty * output.num_steps.float()
        
        # Value target is the actual return (reward)
        # For now, we use the final reward as the return for all steps
        # (could do discounted returns, but single-step reward is simpler)
        value_target = reward.unsqueeze(1).expand(-1, output.values.shape[1])
        
        # Value loss: MSE between predicted and actual
        # Only for steps that were actually taken
        step_mask = torch.arange(output.values.shape[1], device=output.values.device)
        step_mask = step_mask.unsqueeze(0) < output.num_steps.unsqueeze(1)  # (batch, max_steps)
        
        value_loss = F.mse_loss(
            output.values * step_mask.float(),
            value_target.detach() * step_mask.float(),
            reduction="sum",
        ) / (step_mask.sum() + 1e-8)
        
        # Policy loss: REINFORCE with baseline
        # advantage = reward - value (but we use value_target for simplicity)
        advantage = (reward - output.values[:, 0]).detach()  # Use first value as baseline
        
        # Sum log probs over steps taken
        log_prob_sum = (output.log_probs * step_mask.float()).sum(dim=1)
        policy_loss = -(log_prob_sum * advantage).mean()
        
        # Entropy bonus (encourage exploration)
        entropy = (output.entropies * step_mask.float()).sum(dim=1).mean()
        
        # Total loss
        # We also minimize CE on final output to ensure we learn good representations
        ce_loss_mean = ce_loss_final.mean()
        total_loss = (
            ce_loss_mean + 
            policy_loss + 
            self.config.value_coef * value_loss - 
            self.config.entropy_coef * entropy
        )
        
        # Compute accuracies
        pred_base = output.logits_base.argmax(dim=-1)
        pred_final = output.logits_final.argmax(dim=-1)
        
        correct_base = (pred_base == targets)
        correct_final = (pred_final == targets)
        
        local_mask = (modes == 0)
        global_mask = (modes == 1)
        second_min_mask = (modes == 2)
        
        # Layer usage statistics
        layer_counts = {}
        for b in range(self.model.n_blocks + 1):  # Include STOP
            count = (output.actions == b).float().sum().item()
            layer_counts[b] = count
        
        metrics = {
            "ce_loss_base": ce_loss_base.mean().item(),
            "ce_loss_final": ce_loss_final.mean().item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "reward": reward.mean().item(),
            
            "acc_base": correct_base.float().mean().item(),
            "acc_final": correct_final.float().mean().item(),
            "acc_local_base": correct_base[local_mask].float().mean().item() if local_mask.any() else 0.0,
            "acc_local_final": correct_final[local_mask].float().mean().item() if local_mask.any() else 0.0,
            "acc_global_base": correct_base[global_mask].float().mean().item() if global_mask.any() else 0.0,
            "acc_global_final": correct_final[global_mask].float().mean().item() if global_mask.any() else 0.0,
            "acc_second_min_base": correct_base[second_min_mask].float().mean().item() if second_min_mask.any() else 0.0,
            "acc_second_min_final": correct_final[second_min_mask].float().mean().item() if second_min_mask.any() else 0.0,
            
            "avg_steps": output.num_steps.float().mean().item(),
            "avg_steps_local": output.num_steps[local_mask].float().mean().item() if local_mask.any() else 0.0,
            "avg_steps_global": output.num_steps[global_mask].float().mean().item() if global_mask.any() else 0.0,
            "avg_steps_second_min": output.num_steps[second_min_mask].float().mean().item() if second_min_mask.any() else 0.0,
            
            "layer_usage": layer_counts,
            "n_local": local_mask.sum().item(),
            "n_global": global_mask.sum().item(),
            "n_second_min": second_min_mask.sum().item(),
        }
        
        return total_loss, metrics
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Metrics:
        """Train for one epoch."""
        self.model.train()
        
        metrics = Metrics()
        layer_usage_total = {i: 0.0 for i in range(self.model.n_blocks + 1)}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            modes = batch["mode"].to(self.device)
            
            # Forward
            output = self.model(x, deterministic=False)
            
            # Compute loss
            loss, batch_metrics = self.compute_loss(output, y, modes)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.grad_clip
                )
            
            self.optimizer.step()
            
            # Accumulate metrics
            batch_size = x.shape[0]
            metrics.n_samples += batch_size
            metrics.n_local += batch_metrics["n_local"]
            metrics.n_global += batch_metrics["n_global"]
            metrics.n_second_min += batch_metrics["n_second_min"]
            
            metrics.total_loss += loss.item() * batch_size
            metrics.ce_loss_base += batch_metrics["ce_loss_base"] * batch_size
            metrics.ce_loss_final += batch_metrics["ce_loss_final"] * batch_size
            metrics.policy_loss += batch_metrics["policy_loss"] * batch_size
            metrics.value_loss += batch_metrics["value_loss"] * batch_size
            metrics.entropy += batch_metrics["entropy"] * batch_size
            
            metrics.acc_base += batch_metrics["acc_base"] * batch_size
            metrics.acc_final += batch_metrics["acc_final"] * batch_size
            metrics.acc_local_base += batch_metrics["acc_local_base"] * batch_metrics["n_local"]
            metrics.acc_local_final += batch_metrics["acc_local_final"] * batch_metrics["n_local"]
            metrics.acc_global_base += batch_metrics["acc_global_base"] * batch_metrics["n_global"]
            metrics.acc_global_final += batch_metrics["acc_global_final"] * batch_metrics["n_global"]
            metrics.acc_second_min_base += batch_metrics["acc_second_min_base"] * batch_metrics["n_second_min"]
            metrics.acc_second_min_final += batch_metrics["acc_second_min_final"] * batch_metrics["n_second_min"]
            
            metrics.avg_steps += batch_metrics["avg_steps"] * batch_size
            metrics.avg_steps_local += batch_metrics["avg_steps_local"] * batch_metrics["n_local"]
            metrics.avg_steps_global += batch_metrics["avg_steps_global"] * batch_metrics["n_global"]
            metrics.avg_steps_second_min += batch_metrics["avg_steps_second_min"] * batch_metrics["n_second_min"]
            
            for k, v in batch_metrics["layer_usage"].items():
                layer_usage_total[k] += v
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "acc": f"{batch_metrics['acc_final']:.3f}",
                "steps": f"{batch_metrics['avg_steps']:.2f}",
            })
        
        # Average metrics
        n = metrics.n_samples
        n_local = max(metrics.n_local, 1)
        n_global = max(metrics.n_global, 1)
        n_second_min = max(metrics.n_second_min, 1)
        
        metrics.total_loss /= n
        metrics.ce_loss_base /= n
        metrics.ce_loss_final /= n
        metrics.policy_loss /= n
        metrics.value_loss /= n
        metrics.entropy /= n
        
        metrics.acc_base /= n
        metrics.acc_final /= n
        metrics.acc_local_base /= n_local
        metrics.acc_local_final /= n_local
        metrics.acc_global_base /= n_global
        metrics.acc_global_final /= n_global
        metrics.acc_second_min_base /= n_second_min
        metrics.acc_second_min_final /= n_second_min
        
        metrics.avg_steps /= n
        metrics.avg_steps_local /= n_local
        metrics.avg_steps_global /= n_global
        metrics.avg_steps_second_min /= n_second_min
        
        # Normalize layer usage
        total_actions = sum(layer_usage_total.values())
        metrics.layer_usage = {
            k: v / total_actions if total_actions > 0 else 0.0
            for k, v in layer_usage_total.items()
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
    ) -> Metrics:
        """Evaluate on a dataset."""
        self.model.eval()
        
        metrics = Metrics()
        layer_usage_total = {i: 0.0 for i in range(self.model.n_blocks + 1)}
        
        for batch in data_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            modes = batch["mode"].to(self.device)
            
            # Forward (deterministic)
            output = self.model(x, deterministic=True)
            
            # Compute metrics (no loss backward)
            _, batch_metrics = self.compute_loss(output, y, modes)
            
            # Accumulate
            batch_size = x.shape[0]
            metrics.n_samples += batch_size
            metrics.n_local += batch_metrics["n_local"]
            metrics.n_global += batch_metrics["n_global"]
            metrics.n_second_min += batch_metrics["n_second_min"]
            
            metrics.ce_loss_base += batch_metrics["ce_loss_base"] * batch_size
            metrics.ce_loss_final += batch_metrics["ce_loss_final"] * batch_size
            
            metrics.acc_base += batch_metrics["acc_base"] * batch_size
            metrics.acc_final += batch_metrics["acc_final"] * batch_size
            metrics.acc_local_base += batch_metrics["acc_local_base"] * batch_metrics["n_local"]
            metrics.acc_local_final += batch_metrics["acc_local_final"] * batch_metrics["n_local"]
            metrics.acc_global_base += batch_metrics["acc_global_base"] * batch_metrics["n_global"]
            metrics.acc_global_final += batch_metrics["acc_global_final"] * batch_metrics["n_global"]
            metrics.acc_second_min_base += batch_metrics["acc_second_min_base"] * batch_metrics["n_second_min"]
            metrics.acc_second_min_final += batch_metrics["acc_second_min_final"] * batch_metrics["n_second_min"]
            
            metrics.avg_steps += batch_metrics["avg_steps"] * batch_size
            metrics.avg_steps_local += batch_metrics["avg_steps_local"] * batch_metrics["n_local"]
            metrics.avg_steps_global += batch_metrics["avg_steps_global"] * batch_metrics["n_global"]
            metrics.avg_steps_second_min += batch_metrics["avg_steps_second_min"] * batch_metrics["n_second_min"]
            
            for k, v in batch_metrics["layer_usage"].items():
                layer_usage_total[k] += v
        
        # Average metrics
        n = metrics.n_samples
        n_local = max(metrics.n_local, 1)
        n_global = max(metrics.n_global, 1)
        n_second_min = max(metrics.n_second_min, 1)
        
        metrics.ce_loss_base /= n
        metrics.ce_loss_final /= n
        
        metrics.acc_base /= n
        metrics.acc_final /= n
        metrics.acc_local_base /= n_local
        metrics.acc_local_final /= n_local
        metrics.acc_global_base /= n_global
        metrics.acc_global_final /= n_global
        metrics.acc_second_min_base /= n_second_min
        metrics.acc_second_min_final /= n_second_min
        
        metrics.avg_steps /= n
        metrics.avg_steps_local /= n_local
        metrics.avg_steps_global /= n_global
        metrics.avg_steps_second_min /= n_second_min
        
        # Normalize layer usage
        total_actions = sum(layer_usage_total.values())
        metrics.layer_usage = {
            k: v / total_actions if total_actions > 0 else 0.0
            for k, v in layer_usage_total.items()
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> List[Dict]:
        """
        Full training loop.
        
        Returns:
            List of metrics dicts per epoch
        """
        history = []
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters()}")
        print()
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Log
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - Loss: {train_metrics.total_loss:.4f}, "
                  f"Acc: {train_metrics.acc_final:.4f} "
                  f"(base: {train_metrics.acc_base:.4f})")
            print(f"  Val   - CE: {val_metrics.ce_loss_final:.4f}, "
                  f"Acc: {val_metrics.acc_final:.4f} "
                  f"(base: {val_metrics.acc_base:.4f})")
            print(f"  Steps - Train: {train_metrics.avg_steps:.2f}, "
                  f"Val: {val_metrics.avg_steps:.2f}")
            print(f"  Local     - Acc: {val_metrics.acc_local_final:.4f}, "
                  f"Steps: {val_metrics.avg_steps_local:.2f}")
            print(f"  Global    - Acc: {val_metrics.acc_global_final:.4f}, "
                  f"Steps: {val_metrics.avg_steps_global:.2f}")
            if val_metrics.n_second_min > 0:
                print(f"  SecondMin - Acc: {val_metrics.acc_second_min_final:.4f}, "
                      f"Steps: {val_metrics.avg_steps_second_min:.2f}")
            
            # Layer usage
            print(f"  Layer usage: ", end="")
            for i in range(self.model.n_blocks):
                print(f"L{i}: {val_metrics.layer_usage.get(i, 0):.1%} ", end="")
            print(f"STOP: {val_metrics.layer_usage.get(self.model.stop_action, 0):.1%}")
            print()
            
            history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            })
        
        return history

