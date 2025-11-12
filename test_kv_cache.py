"""
Test KV cache implementation for correctness and performance.

This script:
1. Loads or trains a small Shakespeare model
2. Generates text with and without KV cache
3. Verifies outputs are identical
4. Measures speedup from caching

For debugging, use debug_kv_model.py:
    import os
    os.environ["KV_DEBUG"] = "1"
    from debug_kv_model import GPT, GPTConfig
"""

import sys
import time
import pickle
from pathlib import Path

import torch

from model import GPTConfig, GPT

# Configuration
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
DATA_DIR = Path("data/shakespeare_char")
OUT_DIR = Path("out-shakespeare-char")
CKPT_PATH = OUT_DIR / "ckpt.pt"

# Test parameters
TEST_TOKENS = 200  # Generate this many tokens
TEMPERATURE = 0.8
TOP_K = 200


def train_model_if_needed():
    """Train a small model if checkpoint doesn't exist."""
    if CKPT_PATH.exists():
        print(f"✓ Found existing checkpoint at {CKPT_PATH}")
        return

    print(f"✗ No checkpoint found at {CKPT_PATH}")
    print("  Training a small model (this will take a few minutes)...")
    print(
        "  Run: python train.py config/train_shakespeare_char.py --device={DEVICE} \\"
    )
    print("       --compile=False --max_iters=1000 --block_size=128 --batch_size=12 \\")
    print("       --n_layer=4 --n_head=4 --n_embd=128 --dropout=0.0")
    sys.exit(1)


def load_model():
    """Load trained model from checkpoint."""
    print(f"Loading model from {CKPT_PATH}...")
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)

    # Create model
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    # Load state dict (handle torch.compile prefix)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    print(f"✓ Model loaded (block_size={gptconf.block_size})")
    return model, gptconf


def load_tokenizer():
    """Load character-level tokenizer."""
    meta_path = DATA_DIR / "meta.pkl"
    if not meta_path.exists():
        print(f"✗ Tokenizer not found at {meta_path}")
        print("  Run: python data/shakespeare_char/prepare.py")
        sys.exit(1)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    stoi = meta["stoi"]
    itos = meta["itos"]

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return "".join([itos[i] for i in l])

    return encode, decode


def test_kv_cache(model, config, encode, decode):
    """Test KV cache correctness and performance."""
    print("\n" + "=" * 80)
    print("TESTING KV CACHE")
    print("=" * 80)

    # Prepare input
    start_text = "ROMEO:"
    start_ids = encode(start_text)
    x = torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...]

    print(f"\nPrompt: '{start_text}'")
    print(f"Generating {TEST_TOKENS} tokens...")
    print(f"Block size: {config.block_size}")

    # Generate WITHOUT cache
    print("\n[1/2] Generating without cache...")
    torch.manual_seed(42)
    t1 = time.time()
    with torch.no_grad():
        y_no_cache = model.generate(
            x,
            max_new_tokens=TEST_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            use_cache=False,
        )
    t_no_cache = time.time() - t1
    output_no_cache = decode(y_no_cache[0].tolist())
    print(f"      Time: {t_no_cache:.2f}s")

    # Generate WITH cache
    print("[2/2] Generating with cache...")
    torch.manual_seed(42)  # Same seed for deterministic comparison
    t1 = time.time()
    with torch.no_grad():
        y_with_cache = model.generate(
            x,
            max_new_tokens=TEST_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            use_cache=True,
        )
    t_with_cache = time.time() - t1
    output_with_cache = decode(y_with_cache[0].tolist())
    print(f"      Time: {t_with_cache:.2f}s")

    # Compare outputs
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if output_no_cache == output_with_cache:
        print("✅ SUCCESS: Outputs are IDENTICAL!")
        print(f"   Generated {len(y_no_cache[0])} tokens successfully.")
        print("\n   Performance:")
        print(f"   - Without cache: {t_no_cache:.2f}s")
        print(f"   - With cache:    {t_with_cache:.2f}s")
        print(f"   - Speedup:       {t_no_cache / t_with_cache:.2f}x")

        if TEST_TOKENS > config.block_size:
            print(
                f"\n   Note: Cache is only used for the first {config.block_size} tokens."
            )
            print(
                f"         After that, both methods process {config.block_size} tokens per step."
            )

        print("\n" + "=" * 80)
        print("SAMPLE OUTPUT")
        print("=" * 80)
        print(output_with_cache[:400])
        print("=" * 80)

        return True
    else:
        print("❌ FAILURE: Outputs are DIFFERENT!")
        print("\n   Finding first difference...")

        # Find first character difference
        min_len = min(len(output_no_cache), len(output_with_cache))
        for i in range(min_len):
            if output_no_cache[i] != output_with_cache[i]:
                print(f"   First difference at character {i}:")
                start = max(0, i - 30)
                end = min(len(output_no_cache), i + 30)
                print(f"   Without cache: ...{output_no_cache[start:end]}...")
                print(f"   With cache:    ...{output_with_cache[start:end]}...")
                break

        # Find first token difference
        tokens_no_cache = y_no_cache[0].tolist()
        tokens_with_cache = y_with_cache[0].tolist()
        for i in range(min(len(tokens_no_cache), len(tokens_with_cache))):
            if tokens_no_cache[i] != tokens_with_cache[i]:
                print(f"\n   First token difference at position {i}:")
                print(
                    f"   Without cache: token {tokens_no_cache[i]} = '{decode([tokens_no_cache[i]])}'"
                )
                print(
                    f"   With cache:    token {tokens_with_cache[i]} = '{decode([tokens_with_cache[i]])}'"
                )
                break

        print("\n   To debug, use debug_kv_model.py:")
        print("   import os")
        print("   os.environ['KV_DEBUG'] = '1'")
        print("   os.environ['KV_DEBUG_LAYER'] = '0'")
        print("   from debug_kv_model import GPT, GPTConfig")
        print("=" * 80)

        return False


def main():
    """Main test function."""
    print("=" * 80)
    print("KV CACHE TEST")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    # Check for trained model
    train_model_if_needed()

    # Load model and tokenizer
    model, config = load_model()
    encode, decode = load_tokenizer()

    # Run test
    success = test_kv_cache(model, config, encode, decode)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
