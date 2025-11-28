"""
Toy dataset for MuThought Level 2 experiments.

Three tasks based on mode token:
- MODE_LOCAL (10): target = last digit (easy, local copy)
- MODE_GLOBAL (11): target = max(digits) (harder, requires global attention)
- MODE_SECOND_MIN (12): target = second smallest digit (hardest, requires sorting)
"""

import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


# Special tokens
MODE_LOCAL = 10       # Easy task: copy last digit
MODE_GLOBAL = 11      # Medium task: max digit
MODE_SECOND_MIN = 12  # Hard task: second smallest digit
VOCAB_SIZE = 13       # Digits 0-9 + 3 mode tokens

# Mode definitions: (token, mode_flag, target_fn)
# This makes it easy to add new modes in the future
MODES: List[Tuple[int, int, callable]] = [
    (MODE_LOCAL, 0, lambda digits: digits[-1]),                    # Last digit
    (MODE_GLOBAL, 1, lambda digits: max(digits)),                  # Max digit
    (MODE_SECOND_MIN, 2, lambda digits: sorted(digits)[1]),        # Second smallest
]


class ToySequenceDataset(Dataset):
    """
    A toy dataset for testing adaptive computation.
    
    Each sample is a sequence of [mode_token, d1, d2, d3, d4, d5] where:
    - mode_token is MODE_LOCAL (10), MODE_GLOBAL (11), or MODE_SECOND_MIN (12)
    - d1-d5 are random digits 0-9
    
    Target depends on mode:
    - MODE_LOCAL: target = d5 (last digit, easy - just copy)
    - MODE_GLOBAL: target = max(d1, d2, d3, d4, d5) (medium - must look at all)
    - MODE_SECOND_MIN: target = second smallest digit (hardest - requires sorting)
    
    The hypothesis is that the model should learn to spend more thinking
    steps on harder inputs since they require more complex computation.
    """
    
    def __init__(
        self,
        n_samples: int,
        mode_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        seq_digits: int = 5,
        seed: int = 42,
    ):
        """
        Args:
            n_samples: Number of samples to generate
            mode_weights: Weights for (local, global, second_min) modes.
                         These are normalized internally to probabilities.
                         Default (1, 1, 1) gives uniform distribution.
            seq_digits: Number of digits in the sequence (default 5)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.n_samples = n_samples
        self.seq_digits = seq_digits
        
        # Normalize weights to probabilities
        total_weight = sum(mode_weights)
        self.mode_probs = [w / total_weight for w in mode_weights]
        
        # Compute cumulative probabilities for sampling
        self.cum_probs = []
        cumsum = 0.0
        for p in self.mode_probs:
            cumsum += p
            self.cum_probs.append(cumsum)
        
        # Set seed for reproducibility
        random.seed(seed)
        
        self.inputs = []
        self.targets = []
        self.modes = []  # 0 = local (easy), 1 = global (medium), 2 = second_min (hard)
        
        for _ in range(n_samples):
            # Generate random digits
            digits = [random.randint(0, 9) for _ in range(seq_digits)]
            
            # Choose mode based on cumulative probabilities
            r = random.random()
            mode_idx = 0
            for i, cum_p in enumerate(self.cum_probs):
                if r < cum_p:
                    mode_idx = i
                    break
            
            mode_token, mode_flag, target_fn = MODES[mode_idx]
            target = target_fn(digits)
            
            # Input = [mode_token] + digits
            x = [mode_token] + digits
            
            self.inputs.append(x)
            self.targets.append(target)
            self.modes.append(mode_flag)
        
        # Convert to tensors
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        self.modes = torch.tensor(self.modes, dtype=torch.long)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.inputs[idx],      # (seq_len,) = (6,) by default
            "y": self.targets[idx],     # scalar target
            "mode": self.modes[idx],    # 0 = local, 1 = global, 2 = second_min (for analysis)
        }


def create_dataloaders(
    train_samples: int = 100_000,
    val_samples: int = 10_000,
    test_samples: int = 10_000,
    batch_size: int = 128,
    mode_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_workers: int = 0,
):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        batch_size: Batch size for dataloaders
        mode_weights: Weights for (local, global, second_min) modes.
                     These are normalized internally to probabilities.
                     Default (1, 1, 1) gives uniform distribution.
        num_workers: Number of dataloader workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    train_ds = ToySequenceDataset(train_samples, mode_weights=mode_weights, seed=42)
    val_ds = ToySequenceDataset(val_samples, mode_weights=mode_weights, seed=123)
    test_ds = ToySequenceDataset(test_samples, mode_weights=mode_weights, seed=999)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size * 2, 
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size * 2, 
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader

