"""
MuThought Level 2 Model Architecture.

The model treats the "middle" of a language model as a decision process:
- Encoder: Tokens → latent "thought state" (s_0)
- Thought Pool: Set of residual blocks that can be applied in any order
- Policy/Value Heads: Decide which block to apply next (or stop)
- Decoder: Final thought state → output logits
"""

import math
from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .dataset import VOCAB_SIZE


class ThoughtOutput(NamedTuple):
    """Output from a forward pass through MuThought."""
    logits_base: torch.Tensor      # (batch, vocab) - zero-thought logits
    logits_final: torch.Tensor     # (batch, vocab) - after thinking
    log_probs: torch.Tensor        # (batch, max_steps) - log probs of actions taken
    values: torch.Tensor           # (batch, max_steps) - value estimates
    num_steps: torch.Tensor        # (batch,) - actual steps taken per sample
    actions: torch.Tensor          # (batch, max_steps) - actions taken
    entropies: torch.Tensor        # (batch, max_steps) - entropy at each step


class ThoughtBlock(nn.Module):
    """
    A single thought block (residual transformer layer).
    
    This wraps a transformer encoder layer and adds:
    - Pre-norm for stability
    - Residual connection
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, 
            n_heads, 
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply thought block.
        
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            (batch, seq_len, d_model)
        """
        # Pre-norm attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        
        # Pre-norm FFN with residual
        x = x + self.ff(self.norm2(x))
        
        return x


class MuThought(nn.Module):
    """
    MuThought Level 2: Adaptive computation via RL routing.
    
    Architecture:
    - Token + positional embeddings → initial thought state s_0
    - Thought loop:
        - Policy head chooses: apply block_i or STOP
        - If block chosen: s_{t+1} = block_i(s_t + step_emb + block_emb)
        - If STOP: exit loop
    - Decoder head: final state → logits
    
    Training:
    - Zero-thought baseline: decode s_0 directly
    - Thinking path: decode final state after thought loop
    - Reward: improvement in cross-entropy minus step penalty
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        n_blocks: int = 4,
        n_heads: int = 4,
        max_steps: int = 6,
        max_seq_len: int = 16,
        dropout: float = 0.1,
        pool_mode: str = "mean",
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Hidden dimension
            n_blocks: Number of thought blocks in the pool
            n_heads: Number of attention heads per block
            max_steps: Maximum thinking steps allowed
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            pool_mode: How to pool sequence before decoding ("mean" or "last")
        """
        super().__init__()
        
        assert pool_mode in ("mean", "last"), f"pool_mode must be 'mean' or 'last', got {pool_mode}"
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.max_steps = max_steps
        self.stop_action = n_blocks  # STOP action index
        self.pool_mode = pool_mode
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Step embedding: tells the block "what time it is" in thinking
        self.step_emb = nn.Embedding(max_steps, d_model)
        
        # Block ID embedding: tells the block "which block am I"
        self.block_emb = nn.Embedding(n_blocks, d_model)
        
        # Thought pool: N residual blocks
        self.thought_blocks = nn.ModuleList([
            ThoughtBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])
        
        # Normalization before policy/value/decode
        self.norm = nn.LayerNorm(d_model)
        
        # Policy head: outputs distribution over [block_0, ..., block_N-1, STOP]
        # We pool the sequence to get a single vector for policy decision
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_blocks + 1),  # +1 for STOP action
        )
        
        # Value head: predicts expected improvement
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        
        # Decoder head: thought state → logits
        self.decoder = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input tokens to initial thought state.
        
        Args:
            x: (batch, seq_len) input token ids
            
        Returns:
            (batch, seq_len, d_model) initial thought state s_0
        """
        batch_size, seq_len = x.shape
        
        # Token embeddings
        tok_emb = self.token_emb(x)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_emb(positions)
        
        # Combine
        return tok_emb + pos_emb
    
    def decode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Decode thought state to logits.
        
        Args:
            state: (batch, seq_len, d_model)
            
        Returns:
            (batch, vocab_size) logits
        """
        # Pool based on mode
        if self.pool_mode == "mean":
            pooled = state.mean(dim=1)  # (batch, d_model)
        else:  # "last"
            pooled = state[:, -1, :]  # (batch, d_model)
        
        pooled = self.norm(pooled)
        
        # Project to vocab
        return self.decoder(pooled)
    
    def get_policy_value(self, state: torch.Tensor) -> tuple:
        """
        Get policy distribution and value estimate from current state.
        
        Args:
            state: (batch, seq_len, d_model)
            
        Returns:
            policy_logits: (batch, n_blocks + 1)
            value: (batch,)
        """
        # Pool over sequence
        pooled = state[:,-1,:] # (batch, d_model)
        pooled = self.norm(pooled)
        
        # Policy and value
        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled).squeeze(-1)
        
        return policy_logits, value
    
    def apply_block(
        self, 
        state: torch.Tensor, 
        block_idx: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """
        Apply selected thought blocks to state (efficient parallel version).
        Works on CUDA, MPS, and CPU.
        
        Only computes blocks that are actually needed, but runs them in parallel.
        
        Args:
            state: (batch, seq_len, d_model)
            block_idx: (batch,) indices of blocks to apply
            step: Current thinking step (for step embedding)
            
        Returns:
            (batch, seq_len, d_model) new state
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Handle STOP actions (no-op)
        stop_mask = (block_idx >= self.n_blocks)
        new_state = state.clone()
        
        if stop_mask.all():
            return new_state
        
        # Add step embedding (same for all)
        step_emb = self.step_emb(torch.tensor(step, device=device))
        state_with_step = state + step_emb
        
        # Group samples by which block they need
        block_outputs = {}
        sample_indices = {}
        
        for block_id in range(self.n_blocks):
            # Find samples that need this block
            mask = (block_idx == block_id)
            if not mask.any():
                continue
            
            # Get the samples and their original indices
            group_state = state_with_step[mask]
            group_indices = torch.where(mask)[0]
            
            # Add block-specific embedding
            block_emb = self.block_emb(torch.tensor(block_id, device=device))
            group_state = group_state + block_emb
            
            # Apply block to this group (batched, so efficient)
            transformed = self.thought_blocks[block_id](group_state)
            
            # Store results with original indices
            block_outputs[block_id] = transformed
            sample_indices[block_id] = group_indices
        
        # Write results back to output tensor
        for block_id, output in block_outputs.items():
            indices = sample_indices[block_id]
            new_state[indices] = output
        
        return new_state
    
    def forward(
        self, 
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> ThoughtOutput:
        """
        Forward pass with thinking loop.
        
        Args:
            x: (batch, seq_len) input token ids
            deterministic: If True, use argmax for action selection (eval mode)
            
        Returns:
            ThoughtOutput with all relevant tensors
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Encode to initial thought state
        state = self.encode(x)  # (batch, seq_len, d_model)
        
        # Zero-thought baseline
        logits_base = self.decode(state)
        
        # Storage for trajectory
        log_probs_list = []
        values_list = []
        actions_list = []
        entropies_list = []
        
        # Track which samples are still "thinking"
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        num_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Thinking loop
        for step in range(self.max_steps):
            # Get policy and value for active samples
            policy_logits, value = self.get_policy_value(state)
            
            # Sample or argmax action
            dist = Categorical(logits=policy_logits)
            if deterministic:
                action = policy_logits.argmax(dim=-1)
            else:
                action = dist.sample()
            
            # Compute log prob and entropy
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            # Store (masked by active)
            log_probs_list.append(log_prob * active.float())
            values_list.append(value * active.float())
            actions_list.append(action)
            entropies_list.append(entropy * active.float())
            
            # Update step count for active samples
            num_steps = num_steps + active.long()
            
            # Check for STOP actions
            stopped = (action == self.stop_action)
            active = active & ~stopped
            
            # If all stopped, exit early
            if not active.any():
                # Pad remaining steps with zeros
                for _ in range(step + 1, self.max_steps):
                    log_probs_list.append(torch.zeros(batch_size, device=device))
                    values_list.append(torch.zeros(batch_size, device=device))
                    actions_list.append(torch.full((batch_size,), self.stop_action, device=device))
                    entropies_list.append(torch.zeros(batch_size, device=device))
                break
            
            # Apply blocks for active samples (with STOP meaning no-op)
            state = self.apply_block(state, action, step)
        
        # Final decode
        logits_final = self.decode(state)
        
        # Stack trajectory tensors
        log_probs = torch.stack(log_probs_list, dim=1)  # (batch, max_steps)
        values = torch.stack(values_list, dim=1)        # (batch, max_steps)
        actions = torch.stack(actions_list, dim=1)      # (batch, max_steps)
        entropies = torch.stack(entropies_list, dim=1)  # (batch, max_steps)
        
        return ThoughtOutput(
            logits_base=logits_base,
            logits_final=logits_final,
            log_probs=log_probs,
            values=values,
            num_steps=num_steps,
            actions=actions,
            entropies=entropies,
        )
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {
            "embeddings": sum(p.numel() for p in [
                self.token_emb.weight, 
                self.pos_emb.weight,
                self.step_emb.weight,
                self.block_emb.weight,
            ]),
            "thought_blocks": sum(
                p.numel() for block in self.thought_blocks for p in block.parameters()
            ),
            "policy_head": sum(p.numel() for p in self.policy_head.parameters()),
            "value_head": sum(p.numel() for p in self.value_head.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "norm": sum(p.numel() for p in self.norm.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts


# =============================================================================
# Comparison Models
# =============================================================================

class BaselineModel(nn.Module):
    """
    Baseline model: Encoder-Decoder only, no residual/thought layers.
    
    This is the simplest possible model - just embed tokens and decode.
    Used as a lower bound for comparison.
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        max_seq_len: int = 16,
        dropout: float = 0.1,
        pool_mode: str = "mean",
    ):
        super().__init__()
        
        assert pool_mode in ("mean", "last"), f"pool_mode must be 'mean' or 'last', got {pool_mode}"
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pool_mode = pool_mode
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Normalization before decode
        self.norm = nn.LayerNorm(d_model)
        
        # Decoder head
        self.decoder = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: embed → decode.
        
        Args:
            x: (batch, seq_len) input token ids
            
        Returns:
            (batch, vocab_size) logits
        """
        batch_size, seq_len = x.shape
        
        # Embed
        tok_emb = self.token_emb(x)
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_emb(positions)
        state = tok_emb + pos_emb
        
        # Pool based on mode
        if self.pool_mode == "mean":
            pooled = state.mean(dim=1)
        else:  # "last"
            pooled = state[:, -1, :]
        pooled = self.norm(pooled)
        
        return self.decoder(pooled)
    
    def count_parameters(self) -> Dict[str, int]:
        counts = {
            "embeddings": self.token_emb.weight.numel() + self.pos_emb.weight.numel(),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "norm": sum(p.numel() for p in self.norm.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts


class TraditionalModel(nn.Module):
    """
    Traditional transformer-style model with fixed-order residual layers.
    
    Layers are always applied in the same order: L0 → L1 → ... → L_{n-1}.
    This is how standard transformers work.
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 16,
        dropout: float = 0.1,
        pool_mode: str = "mean",
    ):
        super().__init__()
        
        assert pool_mode in ("mean", "last"), f"pool_mode must be 'mean' or 'last', got {pool_mode}"
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.pool_mode = pool_mode
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Fixed stack of residual layers
        self.layers = nn.ModuleList([
            ThoughtBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Normalization before decode
        self.norm = nn.LayerNorm(d_model)
        
        # Decoder head
        self.decoder = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: embed → L0 → L1 → ... → decode.
        
        Args:
            x: (batch, seq_len) input token ids
            
        Returns:
            (batch, vocab_size) logits
        """
        batch_size, seq_len = x.shape
        
        # Embed
        tok_emb = self.token_emb(x)
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_emb(positions)
        state = tok_emb + pos_emb
        
        # Apply layers in fixed order
        for layer in self.layers:
            state = layer(state)
        
        # Pool based on mode
        if self.pool_mode == "mean":
            pooled = state.mean(dim=1)
        else:  # "last"
            pooled = state[:, -1, :]
        pooled = self.norm(pooled)
        
        return self.decoder(pooled)
    
    def count_parameters(self) -> Dict[str, int]:
        counts = {
            "embeddings": self.token_emb.weight.numel() + self.pos_emb.weight.numel(),
            "layers": sum(p.numel() for layer in self.layers for p in layer.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "norm": sum(p.numel() for p in self.norm.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts


class RandomOrderModel(nn.Module):
    """
    Random order model: All blocks applied but in random order each forward pass.
    
    This is an ablation to test whether the learned routing in MuThought matters,
    or if any order works equally well. Uses the same blocks as MuThought but
    shuffles them randomly.
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        n_blocks: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 16,
        dropout: float = 0.1,
        pool_mode: str = "mean",
    ):
        super().__init__()
        
        assert pool_mode in ("mean", "last"), f"pool_mode must be 'mean' or 'last', got {pool_mode}"
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.pool_mode = pool_mode
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Step embedding (like MuThought, to indicate position in sequence)
        self.step_emb = nn.Embedding(n_blocks, d_model)
        
        # Block ID embedding
        self.block_emb = nn.Embedding(n_blocks, d_model)
        
        # Pool of blocks (same as MuThought)
        self.blocks = nn.ModuleList([
            ThoughtBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])
        
        # Normalization before decode
        self.norm = nn.LayerNorm(d_model)
        
        # Decoder head
        self.decoder = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: embed → [all blocks in random order] → decode.
        
        Args:
            x: (batch, seq_len) input token ids
            
        Returns:
            (batch, vocab_size) logits
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Embed
        tok_emb = self.token_emb(x)
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.pos_emb(positions)
        state = tok_emb + pos_emb
        
        # Random permutation of block indices
        if self.training:
            order = torch.randperm(self.n_blocks, device=device)
        else:
            # Use fixed order during evaluation for reproducibility
            order = torch.arange(self.n_blocks, device=device)
        
        # Apply all blocks in random order
        for step, block_idx in enumerate(order):
            # Add step and block embeddings (like MuThought)
            step_emb = self.step_emb(torch.tensor(step, device=device))
            block_emb = self.block_emb(block_idx)
            state = state + step_emb + block_emb
            
            # Apply block
            state = self.blocks[block_idx](state)
        
        # Pool based on mode
        if self.pool_mode == "mean":
            pooled = state.mean(dim=1)
        else:  # "last"
            pooled = state[:, -1, :]
        pooled = self.norm(pooled)
        
        return self.decoder(pooled)
    
    def count_parameters(self) -> Dict[str, int]:
        counts = {
            "embeddings": sum(p.numel() for p in [
                self.token_emb.weight,
                self.pos_emb.weight,
                self.step_emb.weight,
                self.block_emb.weight,
            ]),
            "blocks": sum(p.numel() for block in self.blocks for p in block.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "norm": sum(p.numel() for p in self.norm.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts
