"""
One-shot training script for the CertAgent TextCNN risk scorer.

Run this ONCE before starting the Flask server:

    cd backend
    python train_model.py

What it does
------------
1. Downloads the phishing URL dataset from GitHub (cached as phishing_cache.csv).
2. Trains a character-level TextCNN (URL -> char embeddings -> Conv1D -> classifier).
3. Saves weights to backend/weights.pth  (~10 epochs, ~2-5 min on CPU).

After training:
    python app.py
"""

import sys
import os

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _smoke_test():
    """Quick sanity check: encode a URL and verify tensor shape."""
    from services.ml_model import encode_url, PhishingCNN
    import torch

    print("\n-- TextCNN smoke-test --")
    url   = "http://secure-login.paypal-verify.tk/account/confirm.php"
    token = encode_url(url)
    print(f"  URL    : {url}")
    print(f"  Tensor : shape={tuple(token.shape)}, dtype={token.dtype}")
    print(f"  Tokens : {token[:10].tolist()} ...")

    # Forward pass with a dummy batch
    model = PhishingCNN()
    out = model(token.unsqueeze(0))
    prob = torch.sigmoid(out).item()
    print(f"  Untrained P(phishing) = {prob:.3f}  (random weights, expect ~0.5)")
    print()


if __name__ == "__main__":
    print("\nCertAgent -- TextCNN Training Pipeline")
    print("=" * 45)

    _smoke_test()

    from services.ml_model import train
    train(epochs=10, batch_size=256, lr=1e-3)

    print("\nDone! You can now run:  python app.py")
