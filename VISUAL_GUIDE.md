# KV-Cache Visual Guide

## The Problem: Redundant Computation

### Without KV-Cache (Inefficient)

```
Generation Step 1: "Once upon a"
┌─────────────────────────────────────┐
│ Input: ["Once", "upon", "a"]        │
│                                     │
│ Compute:                            │
│   Q for: "Once", "upon", "a"        │
│   K for: "Once", "upon", "a"  ✓     │
│   V for: "Once", "upon", "a"  ✓     │
│                                     │
│ Output: "time"                      │
└─────────────────────────────────────┘

Generation Step 2: "Once upon a time"
┌─────────────────────────────────────┐
│ Input: ["Once", "upon", "a", "time"]│
│                                     │
│ Compute:                            │
│   Q for: "Once", "upon", "a", "time"│
│   K for: "Once", "upon", "a"   ✗    |  WASTE! (already computed)
│   K for: "time"                ✓    │
│   V for: "Once", "upon", "a"   ✗    |  WASTE! (already computed)
│   V for: "time"                ✓    │
│                                     │
│ Output: "there"                     │
└─────────────────────────────────────┘

Generation Step 3: "Once upon a time there"
┌─────────────────────────────────────┐
│ Input: ["Once", "upon", "a",        │
│         "time", "there"]            │
│                                     │
│ Compute:                            │
│   Q for: all 5 tokens               │
│   K for: "Once", "upon", "a"  ✗     |  WASTE!
│   K for: "time"               ✗     |  WASTE!
│   K for: "there"              ✓     │
│   V for: "Once", "upon", "a"  ✗     |  WASTE!
│   V for: "time"               ✗     |  WASTE!
│   V for: "there"              ✓     │
│                                     │
│ Output: "was"                       │
└─────────────────────────────────────┘

Problem: We recompute K,V for old tokens at EVERY step!
```

### With KV-Cache (Efficient)

```
Generation Step 1: "Once upon a"
┌─────────────────────────────────────┐
│ Input: ["Once", "upon", "a"]        │
│                                     │
│ Compute:                            │
│   Q for: "Once", "upon", "a"        │
│   K for: "Once", "upon", "a"  ✓     │
│   V for: "Once", "upon", "a"  ✓     │
│                                     │
│ Cache: K["Once", "upon", "a"]       │
│        V["Once", "upon", "a"]       │
│                                     │
│ Output: "time"                      │
└─────────────────────────────────────┘

Generation Step 2: "Once upon a time"
┌──────────────────────────────────────┐
│ Input: ["time"]  ← Only new token!   │
│                                      │
│ Compute:                             │
│   Q for: "time"                ✓     │
│   K for: "time"                ✓     │
│   V for: "time"                ✓     │
│                                      │
│ Use cached:                          │
│   K["Once", "upon", "a"] from cache  │
│   V["Once", "upon", "a"] from cache  │
│                                      │
│ Concatenate:                         │
│   K = [K_cached, K_new]              │
│   V = [V_cached, V_new]              │
│                                      │
│ Cache: K["Once", "upon", "a", "time"]│
│        V["Once", "upon", "a", "time"]│
│                                      │
│ Output: "there"                      │
└──────────────────────────────────────┘

Generation Step 3: "Once upon a time there"
┌─────────────────────────────────────┐
│ Input: ["there"]  ← Only new token! │
│                                     │
│ Compute:                            │
│   Q for: "there"               ✓    │
│   K for: "there"               ✓    │
│   V for: "there"               ✓    │
│                                     │
│ Use cached:                         │
│   K["Once"..."time"] from cache     │
│   V["Once"..."time"] from cache     │
│                                     │
│ Concatenate:                        │
│   K = [K_cached, K_new]             │
│   V = [V_cached, V_new]             │
│                                     │
│ Cache: K["Once"..."time", "there"]  │
│        V["Once"..."time", "there"]  │
│                                     │
│ Output: "was"                       │
└─────────────────────────────────────┘

Solution: Compute K,V for new tokens only, reuse cached K,V!
```

## Attention Mechanism Visualization

### Standard Attention (No Cache)

```
Step 2: Generate token after "Once upon a"

Input sequence: ["Once", "upon", "a", "time"]
                   ↓       ↓      ↓      ↓
                ┌────────────────────────────┐
                │   Compute Q, K, V for all  │
                └────────────────────────────┘
                   ↓       ↓      ↓      ↓
              Q: [q1,     q2,    q3,    q4]
              K: [k1,     k2,    k3,    k4]
              V: [v1,     v2,    v3,    v4]

Attention matrix (Q @ K^T):
         k1    k2    k3    k4
    q1 [ ✓     ✗     ✗     ✗  ]  "Once" attends to "Once"
    q2 [ ✓     ✓     ✗     ✗  ]  "upon" attends to "Once", "upon"
    q3 [ ✓     ✓     ✓     ✗  ]  "a" attends to "Once", "upon", "a"
    q4 [ ✓     ✓     ✓     ✓  ]  "time" attends to all

    ✓ = allowed (causal mask)
    ✗ = masked (can't attend to future)

Output: Use attention weights to combine V values
```

### Cached Attention

```
Step 2: Generate token after "Once upon a"

Cached from Step 1:
    K_cached: [k1, k2, k3]  ← Already computed!
    V_cached: [v1, v2, v3]  ← Already computed!

New input: ["time"]
              ↓
         ┌─────────┐
         │ Compute │
         │ Q, K, V │
         └─────────┘
              ↓
         Q: [q4]
         K: [k4]
         V: [v4]

Concatenate with cache:
    K_full = [k1, k2, k3, k4] = [K_cached, k4]
    V_full = [v1, v2, v3, v4] = [V_cached, v4]

Attention matrix (Q @ K^T):
         k1    k2    k3    k4
    q4 [ ✓     ✓     ✓     ✓  ]  "time" attends to all

    Only compute 1 row instead of 4!

Output: Use attention weights to combine V values
```

## Memory Layout

### Cache Structure

```
model.generate() maintains:

kv_caches = [
    # Layer 0
    (
        k: [batch, n_heads, seq_len, head_dim],  # Keys
        v: [batch, n_heads, seq_len, head_dim],  # Values
    ),
    # Layer 1
    (
        k: [batch, n_heads, seq_len, head_dim],
        v: [batch, n_heads, seq_len, head_dim],
    ),
    ...
    # Layer N-1
    (
        k: [batch, n_heads, seq_len, head_dim],
        v: [batch, n_heads, seq_len, head_dim],
    ),
]

Example for GPT-2:
- batch = 1
- n_heads = 12
- seq_len = 100 (grows with generation)
- head_dim = 64
- n_layers = 12

Total cache size:
    12 layers × 2 (K,V) × 1 × 12 × 100 × 64 × 2 bytes (fp16)
    = ~3.5 MB
```

## Data Flow Through Model

### Without Cache

```
Token IDs: [1, 2, 3, 4]
    ↓
┌───────────────────────────┐
│ Token Embeddings          │
│ [emb1, emb2, emb3, emb4]  │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│ Position Embeddings       │
│ [pos0, pos1, pos2, pos3]  │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│ Layer 0                   │
│   Attention (all tokens)  │
│   MLP (all tokens)        │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│ Layer 1                   │
│   Attention (all tokens)  │
│   MLP (all tokens)        │
└───────────────────────────┘
    ↓
    ...
    ↓
┌───────────────────────────┐
│ Layer N                   │
│   Attention (all tokens)  │
│   MLP (all tokens)        │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│ LM Head                   │
│ Output logits for token 4 │
└───────────────────────────┘
```

### With Cache (Step 2)

```
Token ID: [4]  ← Only new token!
    ↓
┌───────────────────────────┐
│ Token Embedding           │
│ [emb4]                    │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│ Position Embedding        │
│ [pos3]  ← Position 3!     │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│ Layer 0                   │
│   Attention:              │
│     Q from: [emb4]        │
│     K from: [cached + new]│
│     V from: [cached + new]│
│   MLP: [emb4]             │
│   Cache: updated          │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│ Layer 1                   │
│   Attention:              │
│     Q from: [emb4]        │
│     K from: [cached + new]│
│     V from: [cached + new]│
│   MLP: [emb4]             │
│   Cache: updated          │
└───────────────────────────┘
    ↓
    ...
    ↓
┌───────────────────────────┐
│ Layer N                   │
│   Attention:              │
│     Q from: [emb4]        │
│     K from: [cached + new]│
│     V from: [cached + new]│
│   MLP: [emb4]             │
│   Cache: updated          │
└───────────────────────────┘
    ↓
┌───────────────────────────┐
│ LM Head                   │
│ Output logits for token 4 │
└───────────────────────────┘
```

## Computational Savings

### Token-by-Token Breakdown

```
Generating 5 tokens: "Once upon a time there"

WITHOUT CACHE:
┌──────┬────────────┬──────────────┬─────────────┐
│ Step │ Input Len  │ Compute K,V  │ Cumulative  │
├──────┼────────────┼──────────────┼─────────────┤
│  1   │     3      │      3       │      3      │
│  2   │     4      │      4       │      7      │
│  3   │     5      │      5       │     12      │
│  4   │     6      │      6       │     18      │
│  5   │     7      │      7       │     25      │
└──────┴────────────┴──────────────┴─────────────┘
Total K,V computations: 25

WITH CACHE:
┌──────┬────────────┬──────────────┬─────────────┐
│ Step │ Input Len  │ Compute K,V  │ Cumulative  │
├──────┼────────────┼──────────────┼─────────────┤
│  1   │     3      │      3       │      3      │
│  2   │     1      │      1       │      4      │
│  3   │     1      │      1       │      5      │
│  4   │     1      │      1       │      6      │
│  5   │     1      │      1       │      7      │
└──────┴────────────┴──────────────┴─────────────┘
Total K,V computations: 7

Savings: 25 - 7 = 18 (72% reduction!)
```

### Scaling with Generation Length

```
Computation Cost (K,V operations)

Without Cache: 1 + 2 + 3 + ... + N = N(N+1)/2 ≈ O(N²)
With Cache:    N

For N=100 tokens:
    Without: 100 × 101 / 2 = 5,050 operations
    With:    100 operations
    Speedup: 50.5x (just for K,V computation!)

For N=1000 tokens:
    Without: 1000 × 1001 / 2 = 500,500 operations
    With:    1000 operations
    Speedup: 500.5x (just for K,V computation!)
```

## Position Embeddings: Critical Detail

### Wrong Approach (Bug!)

```
Step 1: Process "Once upon a"
    Positions: [0, 1, 2]  ✓

Step 2: Process "time" with cache
    Positions: [0]  ✗ WRONG!

    "time" should be at position 3, not 0!
```

### Correct Approach

```
Step 1: Process "Once upon a"
    Positions: [0, 1, 2]  ✓
    Cache length: 3

Step 2: Process "time" with cache
    past_length = cache[0][0].size(2) = 3
    Positions: [3]  ✓ CORRECT!

    "time" correctly gets position 3
```

## Summary: Why KV-Cache Works

```
┌─────────────────────────────────────────────────┐
│ Key Insight:                                    │
│                                                 │
│ In autoregressive generation, once we compute   │
│ K and V for a token, they NEVER CHANGE.         │
│                                                 │
│ Why? Because K and V only depend on:            │
│   1. The token's embedding                      │
│   2. The token's position                       │
│   3. Previous layer outputs                     │
│                                                 │
│ All of these are fixed once computed!           │
│                                                 │
│ Only Q (query) for the NEW token needs to       │
│ attend to all previous K,V.                     │
└─────────────────────────────────────────────────┘

Result:
    ✓ Store K,V after computing them once
    ✓ Reuse for all future tokens
    ✓ Only compute K,V for new tokens
    ✓ Massive speedup with minimal memory cost
```

## Visual Comparison: Timeline

```
Without Cache (Generating 4 tokens):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Token 1: ████████████ (process 1 token)
Token 2: ████████████████████████ (process 2 tokens)
Token 3: ████████████████████████████████████ (process 3 tokens)
Token 4: ████████████████████████████████████████████████ (process 4 tokens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total time: ████████████████████████████████████████████████

With Cache (Generating 4 tokens):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Token 1: ████████████ (process 1 token, build cache)
Token 2: ████████████ (process 1 token, use cache)
Token 3: ████████████ (process 1 token, use cache)
Token 4: ████████████ (process 1 token, use cache)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total time: ████████████████████████████████████████████████

Speedup: ~2.5x for just 4 tokens!
         (gets better with more tokens)
```
