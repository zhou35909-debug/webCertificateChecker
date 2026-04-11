"""
Feature extraction for the PhishingMLP model.

12 continuous features, all normalized to roughly [0, 1].

Index  Name                 Description
  0    url_length_norm      len(domain) / 100  (capped at 1.0)
  1    num_dots             dot count (/ 10)
  2    num_hyphens          hyphen count (/ 5)
  3    num_digits           digit count in domain (/ 20)
  4    digit_ratio          digits / len(domain)
  5    subdomain_count      extra subdomains (/ 5)
  6    is_ip_address        1.0 if domain is an IP, else 0.0
  7    has_at_symbol        1.0 if domain contains @, else 0.0
  8    special_char_count   count of {%, =, &, ?, #} (/ 10)
  9    url_entropy          Shannon entropy of domain chars (/ 5, capped at 1.0)
 10    suspicious_tld       1.0 if TLD is in a known free/abused set, else 0.0
 11    ssl_score            1.0 = CA-verified + hostname match, 0.5 = partial, 0.0 = none
"""

import re
import math
from collections import Counter

_IP_RE  = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_SPEC   = re.compile(r"[%=&?#]")

# Free / frequently-abused TLDs seen in phishing datasets
_SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",   # Freenom free TLDs
    "xyz", "top", "click", "link",   # cheap/spam-heavy
    "pw", "cc", "to",
}

NUM_FEATURES = 12


def extract_ml_features(domain: str, cert_info: dict) -> list:
    """
    Return 12 float features for PhishingMLP.

    Parameters
    ----------
    domain    : bare domain (no scheme/path), e.g. "secure-login.paypal-verify.tk"
    cert_info : dict from cert_checker (used for ssl_score)
    """
    d = domain.lower()

    # 0 -- url_length_norm
    url_len = min(len(d) / 100.0, 1.0)

    # 1 -- num_dots
    dots = d.count(".")
    dots_norm = min(dots / 10.0, 1.0)

    # 2 -- num_hyphens
    hyphens = d.count("-")
    hyph_norm = min(hyphens / 5.0, 1.0)

    # 3 -- num_digits
    digits = sum(c.isdigit() for c in d)
    digits_norm = min(digits / 20.0, 1.0)

    # 4 -- digit_ratio
    digit_ratio = digits / max(len(d), 1)

    # 5 -- subdomain_count
    sub_norm = min(max(dots - 1, 0) / 5.0, 1.0)

    # 6 -- is_ip_address
    is_ip = 1.0 if _IP_RE.match(d) else 0.0

    # 7 -- has_at_symbol
    has_at = 1.0 if "@" in d else 0.0

    # 8 -- special_char_count
    spec = len(_SPEC.findall(d))
    spec_norm = min(spec / 10.0, 1.0)

    # 9 -- url_entropy  (Shannon entropy of character distribution)
    entropy = _shannon_entropy(d)
    entropy_norm = min(entropy / 5.0, 1.0)

    # 10 -- suspicious_tld
    tld = d.rsplit(".", 1)[-1] if "." in d else ""
    susp_tld = 1.0 if tld in _SUSPICIOUS_TLDS else 0.0

    # 11 -- ssl_score
    verified = bool(cert_info.get("ca_verified", False))
    match    = bool(cert_info.get("hostname_match", False))
    if verified and match:
        ssl_score = 1.0
    elif verified or match:
        ssl_score = 0.5
    else:
        ssl_score = 0.0

    return [
        url_len, dots_norm, hyph_norm, digits_norm, digit_ratio,
        sub_norm, is_ip, has_at, spec_norm, entropy_norm,
        susp_tld, ssl_score,
    ]


def get_feature_names() -> list:
    return [
        "url_length_norm",
        "num_dots",
        "num_hyphens",
        "num_digits",
        "digit_ratio",
        "subdomain_count",
        "is_ip_address",
        "has_at_symbol",
        "special_char_count",
        "url_entropy",
        "suspicious_tld",
        "ssl_score",
    ]


def _shannon_entropy(s: str) -> float:
    """Shannon entropy of a string (bits per character)."""
    if not s:
        return 0.0
    counts = Counter(s)
    total  = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())
