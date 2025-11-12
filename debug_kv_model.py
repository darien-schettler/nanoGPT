"""
DEBUG VERSION: GPT Language Model with KV-Cache Debug Instrumentation

This is a debug version of model.py that includes extensive debug printing.
Use this for troubleshooting KV cache issues.

For production use, import from model.py instead (no debug overhead).

DEBUG USAGE:
    import os
    os.environ["KV_DEBUG"] = "1"           # Enable debug output
    os.environ["KV_DEBUG_LAYER"] = "0"     # Optional: only show specific layer

    from debug_kv_model import GPT, GPTConfig
    model = GPT(GPTConfig())
    # ... debug output will print during generation

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

KV-CACHE MODIFICATIONS:
This file includes KV (Key-Value) caching for faster autoregressive generation.
Five main changes are marked with "KV-CACHE CHANGE #N" comments:
  #1: CausalSelfAttention - Always register causal mask buffer
  #2: CausalSelfAttention.forward() - Add cache parameter and read/write logic
  #3: Block.forward() - Thread cache through attention layer
  #4: GPT.forward() - Handle position calculation and thread cache through layers
  #5: GPT.generate() - Manage cache lifecycle during generation

See KV_CACHE.md for detailed documentation.
"""

import math
import inspect
from dataclasses import dataclass
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

# Debug controls (set KV_DEBUG=1 to enable; optional KV_DEBUG_LAYER to filter a layer index)
KV_DEBUG = os.getenv("KV_DEBUG", "0") == "1"
KV_DEBUG_LAYER = os.getenv("KV_DEBUG_LAYER")


def _kvdbg_enabled_for_layer(layer_idx: int | None) -> bool:
    if not KV_DEBUG:
        return False
    if KV_DEBUG_LAYER is None:
        return True
    if layer_idx is None:
        return True
    return str(layer_idx) == str(KV_DEBUG_LAYER)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # ========================================================================
        # KV-CACHE CHANGE #1: Always register causal mask buffer
        # WHY: Manual attention masking needed when Q and K have different lengths
        #      (e.g., T_q=1, T_k=128 during cached generation)
        # ========================================================================
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    # ========================================================================
    # KV-CACHE CHANGE #2: Add kv_cache parameter to forward()
    # WHAT: Accept and return cache tuple to enable K/V reuse across generation steps
    # ========================================================================
    def forward(self, x, kv_cache=None):
        """
        Forward pass with optional KV caching.

        Args:
            x (Tensor): Input tensor of shape (B, T, C)
            kv_cache (tuple, optional): Cache tuple (k_cache, v_cache, cache_pos)
                - k_cache: Tensor of shape (B, n_head, block_size, head_dim)
                - v_cache: Tensor of shape (B, n_head, block_size, head_dim)
                - cache_pos: Integer, number of tokens currently cached

        Returns:
            tuple: (output, new_cache)
                - output: Tensor of shape (B, T, C)
                - new_cache: Updated cache tuple
        """
        B, T, C = x.size()  # (batch, seq_len, n_embd)

        # (1) Compute Q, K, V for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # Each: (B, T, C)
        # (1.1) Reshape to separate heads: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # ========================================================================
        # KV-CACHE CHANGE #2 (continued): Cache read/write logic
        # WHAT: Store new K/V in cache, read all cached K/V for attention
        # NOTE: Cache only works up to block_size tokens (no sliding window)
        # ========================================================================
        if kv_cache is not None:
            # (2.1) Unpack cache: k_cache/v_cache are (B, nh, block_size, hs)
            k_cache, v_cache, cache_pos = kv_cache
            max_ctx = self.bias.size(-1)  # block_size
            new_cache_pos = cache_pos + T

            if _kvdbg_enabled_for_layer(getattr(self, "layer_idx", None)):
                print(
                    f"[KVDBG][Attn L{getattr(self, 'layer_idx', '?')}] "
                    f"cache_pos={cache_pos} T={T} new_cache_pos={new_cache_pos} max_ctx={max_ctx}"
                )

            # (2.2) Write new K/V into cache at positions [cache_pos:cache_pos+T]
            k_cache[:, :, cache_pos:new_cache_pos, :] = k  # (B, nh, block_size, hs)
            v_cache[:, :, cache_pos:new_cache_pos, :] = v  # (B, nh, block_size, hs)

            # (2.3) Use all cached K/V (from position 0 to new_cache_pos)
            k = k_cache[:, :, :new_cache_pos, :]  # (B, nh, new_cache_pos, hs)
            v = v_cache[:, :, :new_cache_pos, :]  # (B, nh, new_cache_pos, hs)

            if _kvdbg_enabled_for_layer(getattr(self, "layer_idx", None)):
                print(
                    f"[KVDBG][Attn L{getattr(self, 'layer_idx', '?')}] "
                    f"using cache T_k={k.size(2)}"
                )
        else:
            # (2.4) First pass: allocate and initialize cache
            max_ctx = self.bias.size(-1)  # block_size
            k_cache = torch.zeros(
                B,
                self.n_head,
                max_ctx,
                C // self.n_head,
                dtype=k.dtype,
                device=k.device,
            )  # (B, nh, block_size, hs)
            v_cache = torch.zeros(
                B,
                self.n_head,
                max_ctx,
                C // self.n_head,
                dtype=v.dtype,
                device=v.device,
            )  # (B, nh, block_size, hs)
            # (2.4.1) Write initial K/V
            k_cache[:, :, :T, :] = k
            v_cache[:, :, :T, :] = v
            new_cache_pos = T

            if _kvdbg_enabled_for_layer(getattr(self, "layer_idx", None)):
                print(
                    f"[KVDBG][Attn L{getattr(self, 'layer_idx', '?')}] "
                    f"init_cache T={T} max_ctx={max_ctx}"
                )

        # (2.5) Package updated cache for return
        new_cache = (k_cache, v_cache, new_cache_pos)

        # (3) Compute attention scores: Q @ K^T scaled by sqrt(head_dim)
        att = (q @ k.transpose(-2, -1)) * (
            1.0 / math.sqrt(k.size(-1))
        )  # (B, nh, T_q, T_k)

        # (4) Apply causal mask to prevent attending to future tokens
        T_q = q.size(2)  # Query length (usually 1 during cached generation)
        T_k = k.size(2)  # Key length (grows with cache: 1, 2, ..., block_size)

        if _kvdbg_enabled_for_layer(getattr(self, "layer_idx", None)):
            print(
                f"[KVDBG][Attn L{getattr(self, 'layer_idx', '?')}] mask T_q={T_q} T_k={T_k}"
            )

        if T_q == T_k:
            # (4.1) Standard case: same length (training or first generation step)
            # Use standard causal mask: position i can attend to positions [0..i]
            att = att.masked_fill(self.bias[:, :, :T_q, :T_k] == 0, float("-inf"))
        else:
            # (4.2) Cache case: T_q < T_k (e.g., T_q=1, T_k=128)
            # New query is at the end of the sequence, can attend to all previous keys
            offset = T_k - T_q  # e.g., 128 - 1 = 127
            row_idx = torch.arange(T_q, device=att.device) + offset  # [127]
            causal_mask = self.bias[:, :, row_idx, :T_k]  # (1, 1, T_q, T_k)
            att = att.masked_fill(causal_mask == 0, float("-inf"))

        # (5) Apply softmax and compute weighted sum of values
        att = F.softmax(att, dim=-1)  # (B, nh, T_q, T_k)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T_q, T_k) @ (B, nh, T_k, hs) -> (B, nh, T_q, hs)

        # (6) Concatenate heads and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.resid_dropout(self.c_proj(y))  # (B, T, C)

        return y, new_cache


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    # ========================================================================
    # KV-CACHE CHANGE #3: Thread cache through Block.forward()
    # WHAT: Pass cache to attention layer and return updated cache
    # ========================================================================
    def forward(self, x, kv_cache=None):
        """
        Forward pass through transformer block with optional KV caching.

        Args:
            x (Tensor): Input of shape (B, T, C)
            kv_cache (tuple, optional): KV cache for attention layer

        Returns:
            tuple: (output, new_cache)
                - output: Tensor of shape (B, T, C)
                - new_cache: Updated cache tuple
        """
        # (1) Self-attention with residual connection
        attn_output, new_cache = self.attn(self.ln_1(x), kv_cache=kv_cache)  # (B, T, C)
        x = x + attn_output  # (B, T, C)

        # (2) MLP with residual connection
        x = x + self.mlp(self.ln_2(x))  # (B, T, C)

        return x, new_cache


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying
        # Assign stable layer indices for debugging visibility
        for i, block in enumerate(self.transformer.h):
            setattr(block, "layer_idx", i)
            setattr(block.attn, "layer_idx", i)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ========================================================================
    # KV-CACHE CHANGE #4: Add kv_cache parameter to GPT.forward()
    # WHAT: Compute positions based on cache state, thread cache through all layers
    # ========================================================================
    def forward(self, idx, targets=None, kv_cache=None):
        """
        Forward pass through GPT model with optional KV caching.

        Args:
            idx (Tensor): Input token indices of shape (B, T)
            targets (Tensor, optional): Target token indices of shape (B, T) for loss computation
            kv_cache (list[tuple], optional): List of cache tuples, one per layer

        Returns:
            tuple: (logits, loss, new_caches)
                - logits: Tensor of shape (B, T, vocab_size) or (B, 1, vocab_size) if targets=None
                - loss: Scalar tensor if targets provided, else None
                - new_caches: List of updated cache tuples, one per layer
        """
        device = idx.device
        b, t = idx.size()  # (batch, seq_len)
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        # ========================================================================
        # KV-CACHE CHANGE #4 (continued): Position calculation
        # WHAT: Use cache_pos to determine correct positions for new tokens
        # ========================================================================
        # (1) Compute position indices for position embeddings
        if kv_cache is not None:
            # (1.1) With cache: new tokens start at position cache_pos
            cache_pos = kv_cache[0][2]  # All layers share same position counter
            pos = torch.arange(
                cache_pos, cache_pos + t, dtype=torch.long, device=device
            )  # (T,)
        else:
            # (1.2) Without cache: positions start at 0
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # (T,)
        if KV_DEBUG:
            if kv_cache is None:
                if t <= 16:
                    print(
                        f"[KVDBG][GPT] kv_cache=None b={b} t={t} pos={pos.detach().cpu().tolist()}"
                    )
                else:
                    print(
                        f"[KVDBG][GPT] kv_cache=None b={b} t={t} pos[0]={pos[0].item()} pos[-1]={pos[-1].item()}"
                    )
            else:
                if t <= 16:
                    print(
                        f"[KVDBG][GPT] kv_cache=present b={b} t={t} cache_pos={cache_pos} "
                        f"pos={pos.detach().cpu().tolist()}"
                    )
                else:
                    print(
                        f"[KVDBG][GPT] kv_cache=present b={b} t={t} cache_pos={cache_pos} "
                        f"pos[0]={pos[0].item()} pos[-1]={pos[-1].item()}"
                    )

        # (2) Get token and position embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, n_embd)

        # ========================================================================
        # KV-CACHE CHANGE #4 (continued): Thread cache through all layers
        # WHAT: Each layer gets its own cache, returns updated cache
        # ========================================================================
        # (3) Pass through transformer blocks with KV caching
        new_caches = []  # Store updated cache from each layer
        for i, block in enumerate(self.transformer.h):
            # (3.1) Get cache for this specific layer
            layer_cache = kv_cache[i] if kv_cache is not None else None

            if _kvdbg_enabled_for_layer(getattr(block, "layer_idx", None)):
                print(
                    f"[KVDBG][Block {getattr(block, 'layer_idx', '?')}] "
                    f"x.shape={x.shape} kv_cache={'present' if layer_cache is not None else 'none'}"
                )

            # (3.2) Forward through block, get updated cache
            x, new_layer_cache = block(x, kv_cache=layer_cache)  # (B, T, n_embd)
            new_caches.append(new_layer_cache)

        # (4) Final layer norm
        x = self.transformer.ln_f(x)  # (B, T, n_embd)

        # (5) Compute logits and loss
        if targets is not None:
            # (5.1) Training: compute loss over all positions
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # (5.2) Inference: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss, new_caches

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    # ========================================================================
    # KV-CACHE CHANGE #5: Update generate() to use cache
    # WHAT: Process only last token when cache active, disable cache when full
    # ========================================================================
    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=False
    ):
        """
        Generate tokens autoregressively with optional KV caching.

        Args:
            idx (Tensor): Input token indices of shape (B, T)
            max_new_tokens (int): Number of tokens to generate
            temperature (float, optional): Sampling temperature (default: 1.0)
            top_k (int, optional): If set, only sample from top k tokens
            use_cache (bool, optional): Enable KV caching for speedup (default: False)

        Returns:
            Tensor: Generated sequence of shape (B, T + max_new_tokens)

        Note:
            Cache is only used for the first block_size tokens. After that,
            generation falls back to standard processing.
        """
        kv_cache = None  # Initialize cache

        for step in range(max_new_tokens):
            # (1) Determine if cache is full (reached block_size limit)
            cache_full = (
                kv_cache is not None and kv_cache[0][2] >= self.config.block_size
            )

            # (2) Prepare input for this step
            if use_cache and kv_cache is not None and not cache_full:
                # (2.1) Cache active and not full: process only last token
                idx_cond = idx[:, [-1]]  # (B, 1)
            else:
                # (2.2) No cache or cache full: crop to block_size
                idx_cond = (
                    idx
                    if idx.size(1) <= self.config.block_size
                    else idx[:, -self.config.block_size :]
                )  # (B, T) where T <= block_size
                # (2.2.1) If cache was full, disable it
                if cache_full:
                    kv_cache = None

            if KV_DEBUG:
                print(
                    f"[KVDBG][generate] step={step} use_cache={use_cache} "
                    f"kv_cache_present={kv_cache is not None} cache_full={cache_full} "
                    f"idx_cond.shape={tuple(idx_cond.shape)}"
                )

            # (3) Forward pass (with or without cache)
            logits, _, kv_cache = self(
                idx_cond, kv_cache=kv_cache if (use_cache and not cache_full) else None
            )  # logits: (B, T, vocab_size)

            # (4) Sample next token
            # (4.1) Get logits for last position and apply temperature
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # (4.2) Optionally restrict to top-k tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # (4.3) Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # (5) Append sampled token to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
