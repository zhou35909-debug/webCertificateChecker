"""
PhishingCNN -- character-level TextCNN for phishing URL detection.

Architecture (Kim 2014 / TextCNN adapted to URLs):
  Input  : URL string, padded to MAX_LEN characters
  Embed  : each char -> 16-dim vector
  Conv1D : 3 parallel filters (kernels 3,4,5) x 128 filters each
  MaxPool: global max over sequence length
  Concat : 384-dim (128 * 3)
  FC     : 384 -> 128 -> 1 (sigmoid = P phishing)

Why CNN over MLP:
  MLP operates on hand-crafted features and hits ~65% accuracy.
  CNN learns URL sub-string patterns directly (e.g. "login", "secure",
  ".php?id=", long random tokens) and reaches ~90%+ on URL datasets.

Usage
-----
Training (once):
    python train_model.py

Inference (per request):
    from services.ml_model import load_model, predict_risk
    model = load_model()
    score = predict_risk("evil-login.verify-paypal.tk", model)
"""

import os
import torch
import torch.nn as nn

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "weights.pth")

# -- Character vocabulary -------------------------------------------------
# All chars that commonly appear in URLs (lowercased).
_VOCAB = (
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "-._~:/?#[]@!$&'()*+,;=%"
)
_CHAR2IDX = {c: i + 1 for i, c in enumerate(_VOCAB)}  # 1-indexed; 0 = PAD
_VOCAB_SIZE = len(_VOCAB) + 1   # +1 for padding token
_UNK_IDX   = _VOCAB_SIZE        # index for out-of-vocab chars
_VOCAB_SIZE_TOTAL = _VOCAB_SIZE + 1  # PAD + known + UNK

MAX_LEN    = 200   # URLs longer than this are truncated (covers 99%+ of real URLs)
EMBED_DIM  = 16
NUM_FILTERS = 128
KERNEL_SIZES = [3, 4, 5]


# -- Encoding helper -------------------------------------------------------

def encode_url(url: str) -> torch.Tensor:
    """
    Convert a URL / domain string to a padded integer tensor of shape (MAX_LEN,).
    Characters not in the vocabulary are mapped to UNK_IDX.
    """
    url = url.lower()[:MAX_LEN]
    indices = [_CHAR2IDX.get(c, _UNK_IDX) for c in url]
    indices += [0] * (MAX_LEN - len(indices))  # right-pad with 0
    return torch.tensor(indices, dtype=torch.long)


# -- Model -----------------------------------------------------------------

class PhishingCNN(nn.Module):
    """
    TextCNN phishing classifier operating on raw URL characters.

    forward(x) expects x of shape (batch, MAX_LEN) with integer char indices.
    Returns a (batch,) tensor of raw logits; apply sigmoid for probability.
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            _VOCAB_SIZE_TOTAL, EMBED_DIM, padding_idx=0
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(EMBED_DIM, NUM_FILTERS, kernel_size=k, padding=0)
            for k in KERNEL_SIZES
        ])
        self.dropout = nn.Dropout(0.5)
        total_filters = NUM_FILTERS * len(KERNEL_SIZES)  # 384
        self.fc = nn.Sequential(
            nn.Linear(total_filters, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, MAX_LEN)
        emb = self.embedding(x)          # (batch, MAX_LEN, EMBED_DIM)
        emb = emb.permute(0, 2, 1)       # (batch, EMBED_DIM, MAX_LEN)

        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(emb))    # (batch, NUM_FILTERS, L)
            c = c.max(dim=2).values      # (batch, NUM_FILTERS)  global max-pool
            pooled.append(c)

        out = torch.cat(pooled, dim=1)   # (batch, 384)
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)  # (batch,)


# -- Training --------------------------------------------------------------

def train(epochs: int = 10, batch_size: int = 256, lr: float = 1e-3):
    """
    Train PhishingCNN on the phishing URL dataset and save weights.pth.

    Parameters
    ----------
    epochs     : passes over the training set (10 is usually enough)
    batch_size : larger batches speed up CNN training significantly
    lr         : Adam learning rate
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset, DataLoader
    from services.data_source import load_training_data

    print("=" * 55)
    print("  CertAgent -- TextCNN Phishing Classifier")
    print("=" * 55)

    urls, labels = load_training_data()   # list[str], np.ndarray

    urls_tr, urls_val, y_tr, y_val = train_test_split(
        urls, labels, test_size=0.2, random_state=42, stratify=labels
    )

    class URLDataset(Dataset):
        def __init__(self, url_list, label_arr):
            self.urls   = url_list
            self.labels = torch.tensor(label_arr, dtype=torch.float32)

        def __len__(self):
            return len(self.urls)

        def __getitem__(self, idx):
            return encode_url(self.urls[idx]), self.labels[idx]

    train_loader = DataLoader(
        URLDataset(urls_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        URLDataset(urls_val, y_val), batch_size=batch_size, num_workers=0
    )

    model     = PhishingCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs
    )

    print(f"\nTraining on {len(urls_tr)} URLs, validating on {len(urls_val)}.\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            logits = model(xb)
            loss   = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                probs  = torch.sigmoid(model(xb))
                preds  = (probs >= 0.5).float()
                correct += (preds == yb).sum().item()
                total   += len(yb)
        avg_loss = running_loss / len(train_loader)
        acc      = 100.0 * correct / total
        print(f"  Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  val_acc={acc:.1f}%")

    path = os.path.abspath(_WEIGHTS_PATH)
    torch.save(model.state_dict(), path)
    print(f"\n[OK] Weights saved -> {path}")
    return model


# -- Inference -------------------------------------------------------------

def load_model():
    """Load trained model. Returns None if weights.pth doesn't exist."""
    path = os.path.abspath(_WEIGHTS_PATH)
    if not os.path.exists(path):
        print("[ml_model] weights.pth not found -- run `python train_model.py` first.")
        return None

    model = PhishingCNN()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_risk(url_or_domain: str, model=None) -> float | None:
    """
    Return P(phishing) in [0.0, 1.0] for a URL or bare domain string.

    Parameters
    ----------
    url_or_domain : e.g. "secure-login.paypal-verify.tk" or "https://..."
    model         : pre-loaded PhishingCNN (pass to avoid disk reads per call)

    Returns None if the model has not been trained yet.
    """
    if model is None:
        model = load_model()
    if model is None:
        return None

    x = encode_url(url_or_domain).unsqueeze(0)  # (1, MAX_LEN)
    with torch.no_grad():
        logit = model(x)
        prob  = torch.sigmoid(logit).item()
    return prob
