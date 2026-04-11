"""
Automated data acquisition - URL-based phishing dataset.

Downloads a real phishing URL dataset from GitHub, extracts 5 structural
features directly from each URL string, and returns (X, y) arrays for
PyTorch training.

Features extracted from raw URLs (all use -1 / 0 / 1 encoding):
  0  is_ip_address     -1 = IP, 1 = hostname
  1  url_length         1 = short (<54), 0 = medium, -1 = long (>75)
  2  has_at_symbol     -1 = has @,  1 = no @
  3  has_hyphen        -1 = has -,  1 = no -
  4  subdomain_depth    1 = simple, 0 = one extra level, -1 = deep

Label mapping:
  "phishing" / "bad" / "1" / "-1"  -> 1  (phishing)
  "legitimate" / "good" / "0" / "1" (UCI)  -> 0  (safe)
"""

import io
import math
import os
import re
from collections import Counter
import numpy as np
import pandas as pd
import requests

# Dataset URLs (tried in order until one succeeds)
_DATASET_URLS = [
    # faizann24 - "Fusing RNN+CNN for Malicious URL Detection" (ICLR 2018)
    # Columns: url, isMalicious (0/1)
    "https://raw.githubusercontent.com/faizann24/Fusing-Recurrent-and-Convolutional-Neural-Network-for-Detecting-Malicious-URL/master/data/training_data.csv",
    # incertum/cyber-matrix-ai - mega deep learning URL dataset
    "https://raw.githubusercontent.com/incertum/cyber-matrix-ai/master/Malicious-URL-Detection-Deep-Learning/data/url_data_mega_deep_learning.csv",
    # cahya-wirawan/text-classifier datasets
    "https://raw.githubusercontent.com/cahya-wirawan/text-classifier/master/data/phishing_urls.csv",
]

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "phishing_cache.csv")

# Known URL column names (tried in order)
_URL_COLS = ["url", "URL", "Url", "domain", "Domain", "website", "address"]
_LABEL_COLS = [
    "label", "Label", "class", "Class", "type", "Type",
    "status", "Status", "result", "Result",
    "isMalicious", "is_malicious", "malicious", "Malicious",
]

_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_SPEC  = re.compile(r"[%=&?#]")
_SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "link", "pw", "cc", "to",
}


# Public API

def load_training_data():
    """
    Return (urls, labels) for TextCNN training.
      urls   : list[str]      -- raw URL strings (training input)
      labels : np.ndarray     -- float32, shape (N,); 0=safe, 1=phishing
    """
    df = _load_cached()
    if df is None:
        df = _download()

    if df is not None:
        result = _parse_dataframe(df)
        if result is not None:
            urls, labels = result
            n_p = int(labels.sum())
            print(f"[data_source] Ready -- {len(urls)} URLs "
                  f"({n_p} phishing, {len(urls) - n_p} safe).")
            return urls, labels

    print("[data_source] All downloads failed -- using synthetic data.")
    return _generate_synthetic()


def sample_real_urls(n=10):
    """
    Return a random sample of real URLs from the cached dataset with labels.
    Columns: url, label (0=safe, 1=phishing)
    """
    df = _load_cached()
    if df is None:
        df = _download()
    if df is None:
        print("[data_source] No cached dataset -- run load_training_data() first.")
        return pd.DataFrame()

    url_col   = _find_col(df, _URL_COLS)
    label_col = _find_col(df, _LABEL_COLS)
    if url_col is None or label_col is None:
        print("[data_source] Cannot identify URL/label columns.")
        return pd.DataFrame()

    df = df[[url_col, label_col]].dropna().copy()
    df["label_num"] = df[label_col].apply(_parse_label)
    df = df[df["label_num"].notna()].copy()
    df["label_num"] = df["label_num"].astype(int)

    sample = df.sample(min(n, len(df)), random_state=None).reset_index(drop=True)
    sample = sample.rename(columns={url_col: "url", "label_num": "label"})
    return sample[["url", "label"]]


# Download & cache

def _load_cached():
    path = os.path.abspath(_CACHE_PATH)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"[data_source] Loaded from cache ({len(df)} rows): {path}")
            return df
        except Exception as e:
            print(f"[data_source] Cache read failed: {e}")
    return None


def _download():
    for url in _DATASET_URLS:
        try:
            print(f"[data_source] Downloading...\n  {url}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            path = os.path.abspath(_CACHE_PATH)
            df.to_csv(path, index=False)
            print(f"[data_source] Cached -> {path}  ({len(df)} rows)")
            return df
        except Exception as e:
            print(f"[data_source] Failed ({url[:70]}): {e}")
    return None


# Parsing

def _parse_dataframe(df):
    url_col   = _find_col(df, _URL_COLS)
    label_col = _find_col(df, _LABEL_COLS)

    if url_col is None:
        print(f"[data_source] No URL column found. Columns: {list(df.columns)}")
        return None
    if label_col is None:
        print(f"[data_source] No label column found. Columns: {list(df.columns)}")
        return None

    print(f"[data_source] Using columns: url='{url_col}', label='{label_col}'")

    df = df[[url_col, label_col]].dropna().copy()
    df["_label"] = df[label_col].apply(_parse_label)
    df = df[df["_label"].notna()].copy()

    if len(df) == 0:
        print("[data_source] No valid rows after label parsing.")
        return None

    urls   = df[url_col].astype(str).tolist()
    labels = df["_label"].values.astype(np.float32)
    return urls, labels


def _find_col(df, candidates):
    """Return the first candidate column name that exists in df (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _parse_label(raw):
    """Map diverse label formats to 0.0 (safe) or 1.0 (phishing)."""
    s = str(raw).strip().lower()
    if s in ("phishing", "bad", "malicious", "1", "true"):
        return 1.0
    if s in ("legitimate", "good", "benign", "safe", "0", "false"):
        return 0.0
    # UCI Result column: -1 = phishing, 1 = legitimate
    try:
        v = float(s)
        if v == -1.0:
            return 1.0
        if v == 1.0:
            return 0.0
    except ValueError:
        pass
    return None


# URL feature extraction

def _extract_domain(url):
    """Strip scheme and path, return bare domain."""
    u = url.lower().strip()
    for prefix in ("https://", "http://", "ftp://", "//"):
        if u.startswith(prefix):
            u = u[len(prefix):]
    return u.split("/")[0].split("?")[0].split(":")[0]


def _shannon_entropy(s):
    if not s:
        return 0.0
    counts = Counter(s)
    total  = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _url_to_features(url):
    """
    Extract 12 continuous features from a raw URL string.
    Must stay in sync with feature_extractor.extract_ml_features().
    ssl_score = 0.5 (neutral) since live cert data is unavailable at training.
    """
    domain = _extract_domain(url)
    d = domain.lower()

    url_len     = min(len(d) / 100.0, 1.0)
    dots        = d.count(".")
    dots_norm   = min(dots / 10.0, 1.0)
    hyphens     = d.count("-")
    hyph_norm   = min(hyphens / 5.0, 1.0)
    digits      = sum(c.isdigit() for c in d)
    digits_norm = min(digits / 20.0, 1.0)
    digit_ratio = digits / max(len(d), 1)
    sub_norm    = min(max(dots - 1, 0) / 5.0, 1.0)
    is_ip       = 1.0 if _IP_RE.match(d) else 0.0
    has_at      = 1.0 if "@" in d else 0.0
    spec        = len(_SPEC.findall(url))
    spec_norm   = min(spec / 10.0, 1.0)
    entropy     = min(_shannon_entropy(d) / 5.0, 1.0)
    tld         = d.rsplit(".", 1)[-1] if "." in d else ""
    susp_tld    = 1.0 if tld in _SUSPICIOUS_TLDS else 0.0
    ssl_score   = 0.5

    return [
        url_len, dots_norm, hyph_norm, digits_norm, digit_ratio,
        sub_norm, is_ip, has_at, spec_norm, entropy,
        susp_tld, ssl_score,
    ]


# Synthetic fallback -- generates plausible URL strings for the CNN

_LEGIT_TEMPLATES = [
    "https://www.{word}.com/{path}",
    "https://{word}.org/about",
    "https://{word}.edu/index.html",
    "https://shop.{word}.com/cart",
    "https://{word}.gov/services",
]
_PHISH_TEMPLATES = [
    "http://{ip}/login.php?redirect={word}",
    "http://{word}-secure-verify.{freetld}/update",
    "http://www.{word}.{freetld}/account/confirm.php",
    "http://{num}.{num}.{num}.{num}/{word}/login",
    "http://secure-{word}-{word}.xyz/validate?token={hash}",
    "http://{word}.com.{freetld}/signin/verify.php",
]
_WORDS   = ["paypal", "google", "amazon", "apple", "facebook", "netflix",
            "bank", "secure", "account", "update", "verify", "login"]
_FREETLD = ["tk", "ml", "ga", "cf", "gq", "xyz"]


def _generate_synthetic(n_legit=3000, n_phish=3000):
    import random, hashlib
    rng = random.Random(42)

    def _word():
        return rng.choice(_WORDS)

    def _ip():
        return ".".join(str(rng.randint(1, 254)) for _ in range(4))

    def _hash():
        return hashlib.md5(str(rng.random()).encode()).hexdigest()[:12]

    legit_urls = []
    for _ in range(n_legit):
        tmpl = rng.choice(_LEGIT_TEMPLATES)
        url  = tmpl.format(word=_word(), path=_word())
        legit_urls.append(url)

    phish_urls = []
    for _ in range(n_phish):
        tmpl = rng.choice(_PHISH_TEMPLATES)
        url  = tmpl.format(
            word=_word(), freetld=rng.choice(_FREETLD),
            ip=_ip(), num=str(rng.randint(1, 254)),
            hash=_hash(),
        )
        phish_urls.append(url)

    urls   = legit_urls + phish_urls
    labels = np.array([0.0] * n_legit + [1.0] * n_phish, dtype=np.float32)
    idx    = np.random.default_rng(42).permutation(len(urls))
    urls   = [urls[i] for i in idx]
    labels = labels[idx]
    print(f"[data_source] Synthetic fallback -- {len(urls)} URL strings.")
    return urls, labels
