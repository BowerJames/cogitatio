"""
Toy dataset for MuThought Level 2 experiments.

Three tasks based on mode token:
- MODE_LOCAL (10): target = last digit (easy, local copy)
- MODE_GLOBAL (11): target = max(digits) (harder, requires global attention)
- MODE_SECOND_MIN (12): target = second smallest digit (hardest, requires sorting)
"""

import random
import itertools
from typing import Dict, List, Tuple, Optional

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
        data: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """
        Args:
            n_samples: Number of samples to generate
            mode_weights: Weights for (local, global, second_min) modes.
                         These are normalized internally to probabilities.
                         Default (1, 1, 1) gives uniform distribution.
            seq_digits: Number of digits in the sequence (default 5)
            seed: Random seed for reproducibility
            data: Optional pre-generated data list of dicts {"x": ..., "y": ..., "mode": ...}
                  If provided, n_samples and random generation logic are skipped.
        """
        super().__init__()
        self.seq_digits = seq_digits
        
        if data is not None:
            # Use pre-generated disjoint data
            self.n_samples = len(data)
            self.data = data
        else:
            # Legacy random generation logic
            self.n_samples = n_samples
            
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
            
            # Store in unified format
            self.data = []
            for i in range(n_samples):
                self.data.append({
                    "x": self.inputs[i],
                    "y": self.targets[i],
                    "mode": self.modes[i]
                })
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


def create_dataloaders(
    train_samples: int = 100_000,
    val_samples: int = 10_000,
    test_samples: int = 10_000,
    batch_size: int = 128,
    mode_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_workers: int = 0,
):
    """
    Create train, validation, and test dataloaders with DISJOINT datasets.
    
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
    
    # 1. Generate Universe of all possible unique digit sequences
    # For 5 digits, this is 10^5 = 100,000 unique sequences
    # We will replicate this universe for each mode, effectively giving us
    # 100k * 3 = 300k unique samples (since mode token makes them distinct)
    
    # Generating all 100k tuples is fast
    universe = list(itertools.product(range(10), repeat=5))
    
    # 2. Shuffle universe deterministically
    # We need a stable shuffle so train/val/test splits are consistent
    rng = random.Random(42)
    rng.shuffle(universe)
    
    # 3. Calculate samples needed per mode
    total_samples_needed = train_samples + val_samples + test_samples
    
    # Normalize weights
    total_weight = sum(mode_weights)
    probs = [w / total_weight for w in mode_weights]
    
    # Calculate counts per mode (train, val, test)
    # We allocate proportionally for each split
    def get_counts(total_n, probs):
        counts = [int(total_n * p) for p in probs]
        # Fix rounding errors
        while sum(counts) < total_n:
            counts[0] += 1
        return counts
    
    train_counts = get_counts(train_samples, probs)
    val_counts = get_counts(val_samples, probs)
    test_counts = get_counts(test_samples, probs)
    
    # Check capacity
    max_unique_per_mode = len(universe)
    for mode_idx in range(3):
        needed = train_counts[mode_idx] + val_counts[mode_idx] + test_counts[mode_idx]
        if needed > max_unique_per_mode:
            raise ValueError(
                f"Requested {needed} samples for mode {mode_idx}, but only {max_unique_per_mode} unique sequences exist. "
                "Reduce sample counts or increase sequence length."
            )
            
    # 4. Create datasets
    train_data = []
    val_data = []
    test_data = []
    
    # For each mode, slice the universe
    start_indices = [0, 0, 0]  # Track where we are in the universe for each mode
    
    for mode_idx, (mode_token, mode_flag, target_fn) in enumerate(MODES):
        # Get the universe slice for this mode
        # We use the SAME shuffled universe order for each mode, but that's fine
        # because the mode token makes the inputs distinct ( [10, 1, 2...] != [11, 1, 2...] )
        
        # Train slice
        n_train = train_counts[mode_idx]
        train_digits_list = universe[start_indices[mode_idx] : start_indices[mode_idx] + n_train]
        start_indices[mode_idx] += n_train
        
        # Val slice
        n_val = val_counts[mode_idx]
        val_digits_list = universe[start_indices[mode_idx] : start_indices[mode_idx] + n_val]
        start_indices[mode_idx] += n_val
        
        # Test slice
        n_test = test_counts[mode_idx]
        test_digits_list = universe[start_indices[mode_idx] : start_indices[mode_idx] + n_test]
        start_indices[mode_idx] += n_test
        
        # Helper to convert digits to sample dict
        def make_samples(digits_list):
            samples = []
            for digits in digits_list:
                digits = list(digits)
                target = target_fn(digits)
                x = torch.tensor([mode_token] + digits, dtype=torch.long)
                y = torch.tensor(target, dtype=torch.long)
                m = torch.tensor(mode_flag, dtype=torch.long)
                samples.append({"x": x, "y": y, "mode": m})
            return samples
            
        train_data.extend(make_samples(train_digits_list))
        val_data.extend(make_samples(val_digits_list))
        test_data.extend(make_samples(test_digits_list))
    
    # Shuffle the final datasets so modes are mixed
    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)
    
    train_ds = ToySequenceDataset(len(train_data), data=train_data)
    val_ds = ToySequenceDataset(len(val_data), data=val_data)
    test_ds = ToySequenceDataset(len(test_data), data=test_data)
    
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
