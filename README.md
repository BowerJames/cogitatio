# Cogitatio

**MuThought Level 2**: Adaptive computation in language models via RL-based routing over a pool of thought blocks.

## Overview

Cogitatio implements a novel neural network architecture where the "middle" of a language model (the residual stack) is treated as a **decision process** rather than a fixed pipeline:

- **Encoder**: Maps input tokens into a latent "thought space"
- **Thought Pool**: A set of N residual blocks that can be applied in any order
- **Policy/Value Heads**: Actor-critic heads that decide which block to apply next (or stop)
- **Decoder**: Maps the final thought state back to output logits

Instead of the traditional fixed computation:
```
Embedding → L1 → L2 → ... → L24 → Decode
```

MuThought learns adaptive computation:
```
Embedding → L7 → L3 → L7 → STOP → Decode
```

The **sequence and depth** of computation are chosen dynamically per input by a policy trained with reinforcement learning.

## Key Ideas

1. **Adaptive Compute**: Easy inputs get fewer thinking steps; hard inputs get more
2. **Reusable Thought Skills**: Blocks can specialize and be reused in different orders
3. **RL-Based Routing**: Policy learns which blocks help for which inputs
4. **Zero-Thought Baseline**: Direct embed→decode provides a reference for measuring improvement

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd cogitatio

# Install with uv
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Training

```bash
# Quick test run
uv run python -m cogitatio.scripts.train --epochs 5 --train_samples 10000

# Full training with defaults
uv run python -m cogitatio.scripts.train --epochs 20

# Custom configuration
uv run python -m cogitatio.scripts.train \
    --d_model 128 \
    --n_blocks 6 \
    --max_steps 8 \
    --epochs 30 \
    --step_penalty 0.005
```

### Python API

```python
from cogitatio import MuThought, ToySequenceDataset, MuThoughtTrainer, TrainConfig

# Create model
model = MuThought(
    d_model=64,
    n_blocks=4,
    max_steps=6,
)

# Create dataset
train_ds = ToySequenceDataset(n_samples=100_000)

# Train
config = TrainConfig(epochs=20, step_penalty=0.01)
trainer = MuThoughtTrainer(model, config)
trainer.train(train_loader, val_loader)
```

## Toy Task

The implementation includes a synthetic task designed to test adaptive computation:

| Mode | Token | Task | Difficulty |
|------|-------|------|------------|
| LOCAL | 10 | Return last digit | Easy (local copy) |
| GLOBAL | 11 | Return sum(digits) mod 10 | Hard (global aggregation) |

**Input**: `[mode, d1, d2, d3, d4, d5]` where digits are 0-9

**Hypothesis**: The model should learn to spend more thinking steps on GLOBAL inputs since they require aggregating information across the entire sequence.

## Architecture Details

### Thought Block
Each block is a transformer encoder layer with:
- Multi-head self-attention
- Feed-forward network
- Pre-norm and residual connections

### Step & Block Embeddings
Before applying a block, the state is augmented:
```
s' = s_t + P(t) + B(i)
```
- `P(t)`: Step embedding (tells the block "what time it is" in thinking)
- `B(i)`: Block ID embedding (tells the block "which block am I")

This allows blocks to behave differently early vs late in the thinking process.

### Training Objective
- **Reward**: `R = L_base - L_final - λ * num_steps`
  - Positive reward = thinking helped
  - Step penalty encourages efficiency
- **Policy Loss**: REINFORCE with value baseline
- **Value Loss**: MSE between predicted and actual return
- **Entropy Bonus**: Encourages exploration of different blocks

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 64 | Hidden dimension |
| `n_blocks` | 4 | Number of thought blocks |
| `max_steps` | 6 | Maximum thinking steps |
| `step_penalty` | 0.01 | Cost per thinking step |
| `lr_model` | 1e-3 | LR for embeddings, blocks, decoder |
| `lr_policy` | 3e-3 | LR for policy/value heads (faster) |
| `entropy_coef` | 0.01 | Entropy bonus for exploration |

## Project Structure

```
src/cogitatio/
├── __init__.py      # Package exports
├── dataset.py       # ToySequenceDataset
├── model.py         # MuThought, ThoughtBlock
├── trainer.py       # MuThoughtTrainer, TrainConfig
└── scripts/
    └── train.py     # CLI training script
```

## References

This work is inspired by:
- [MuZero](https://arxiv.org/abs/1911.08265) - Planning with a learned model
- [Universal Transformers](https://arxiv.org/abs/1807.03819) - Adaptive computation time
- [ALBERT](https://arxiv.org/abs/1909.11942) - Weight sharing across layers
- Mixture-of-Experts architectures

## License

MIT

