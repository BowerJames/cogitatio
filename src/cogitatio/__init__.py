"""
Cogitatio: MuThought Level 2 Implementation

A novel architecture for adaptive computation in language models using
RL-based routing over a pool of thought blocks.

Architecture:
- Encoder: Embeds input tokens into "thought space"
- Thought Pool: N residual blocks that can be applied in any order
- Policy/Value Heads: Actor-critic heads that decide which block to apply
- Decoder: Maps final thought state to output logits

Usage:
    from cogitatio import MuThought, ToySequenceDataset, MuThoughtTrainer
    
    # Create model
    model = MuThought(n_blocks=4, max_steps=6)
    
    # Create dataset
    dataset = ToySequenceDataset(n_samples=10000)
    
    # Train
    trainer = MuThoughtTrainer(model, TrainConfig())
    trainer.train(train_loader, val_loader)
"""

from .dataset import (
    ToySequenceDataset,
    create_dataloaders,
    MODE_LOCAL,
    MODE_GLOBAL,
    VOCAB_SIZE,
)
from .model import (
    MuThought,
    ThoughtBlock,
    ThoughtOutput,
    # Comparison models
    BaselineModel,
    TraditionalModel,
    RandomOrderModel,
)
from .trainer import (
    MuThoughtTrainer,
    TrainConfig,
    Metrics,
)

__version__ = "0.1.0"

__all__ = [
    # Dataset
    "ToySequenceDataset",
    "create_dataloaders",
    "MODE_LOCAL",
    "MODE_GLOBAL",
    "VOCAB_SIZE",
    # Model
    "MuThought",
    "ThoughtBlock",
    "ThoughtOutput",
    # Comparison models
    "BaselineModel",
    "TraditionalModel",
    "RandomOrderModel",
    # Training
    "MuThoughtTrainer",
    "TrainConfig",
    "Metrics",
]

