"""
Redirect Chain Inspector — follows a URL through all redirects and
returns structured analysis of the redirect behavior.

Public API
----------
inspect_redirect_chain(url) -> dict
"""

import warnings
from urllib.parse import urlparse

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

_TIMEOUT = 8
_MAX_REDIRECTS = 15


def _parse_domain(url):
    """Return bare hostname from a full URL."""
    try:
        return urlparse(url).netloc.split(":")[0].lower()
    except Exception:
        return ""


def _unavailable(url, reason):
    """Return a structured error result that won't crash the pipeline."""
    return {
        "status": "unavailable",
        "error": reason,
        "input_url": url,
        "final_url": None,
        "redirect_count": None,
        "redirect_chain": [],
        "cross_domain_redirect": None,
        "http_to_https_upgrade": None,
        "significant_destination_change": None,
        "risk_level": "unknown",
        "risk_score": 0.0,
        "evidence": [f"Redirect inspection unavailable: {reason}"],
        "reasoning": f"Could not inspect redirects: {reason}",
    }


def inspect_redirect_chain(url):
    """
    Follow all redirects for a URL and return a structured risk analysis.

    Returns a dict with keys:
      status, input_url, final_url, redirect_count, redirect_chain,
      cross_domain_redirect, http_to_https_upgrade,
      significant_destination_change, risk_level, risk_score,
      evidence, reasoning
    """
    if not url:
        return _unavailable(url, "No URL provided")

    # Normalise — add https:// if scheme is missing
    if "://" not in url:
        url = "https://" + url

    try:
        import requests
    except ImportError:
        return _unavailable(url, "requests library not available")

    try:
        session = requests.Session()
        session.max_redirects = _MAX_REDIRECTS
        resp = session.get(
            url,
            allow_redirects=True,
            timeout=_TIMEOUT,
            verify=False,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CertAgent/1.0)"},
        )
    except requests.exceptions.ConnectionError as e:
        return _unavailable(url, f"Connection error: {_short(e)}")
    except requests.exceptions.Timeout:
        return _unavailable(url, "Request timed out")
    except requests.exceptions.TooManyRedirects:
        return _unavailable(url, f"Too many redirects (>{_MAX_REDIRECTS})")
    except requests.exceptions.SSLError as e:
        return _unavailable(url, f"SSL error: {_short(e)}")
    except Exception as e:
        return _unavailable(url, _short(e))

    # Build the full chain: all intermediate URLs + final URL
    chain = [r.url for r in resp.history] + [resp.url]

    input_url  = chain[0]
    final_url  = chain[-1]
    redirect_count = len(chain) - 1

    input_domain = _parse_domain(input_url)
    final_domain = _parse_domain(final_url)

    cross_domain  = bool(input_domain and final_domain and input_domain != final_domain)
    http_to_https = input_url.startswith("http://") and final_url.startswith("https://")
    significant_change = final_url.rstrip("/") != input_url.rstrip("/")

    # --- Evidence ---
    evidence = []
    if redirect_count == 0:
        evidence.append("No redirects — URL resolves directly.")
    elif redirect_count == 1:
        evidence.append("URL redirected 1 time.")
    else:
        evidence.append(f"URL redirected {redirect_count} times.")

    if http_to_https:
        evidence.append("Connection upgraded from HTTP to HTTPS.")
    if cross_domain:
        evidence.append(
            f"Final destination ({final_domain}) is on a different domain than the input ({input_domain})."
        )
    if significant_change and not cross_domain and redirect_count > 0:
        evidence.append("Final destination URL path differs from the input URL.")

    # --- Risk scoring ---
    score = 0.0
    score += min(redirect_count * 0.12, 0.40)   # more redirects → higher risk
    if cross_domain:
        score += 0.40                             # biggest signal
    if http_to_https and not cross_domain:
        score = max(score - 0.10, 0.0)           # legitimate upgrade slightly reduces risk
    score = min(round(score, 2), 1.0)

    if score >= 0.60:
        risk_level = "high"
    elif score >= 0.25:
        risk_level = "medium"
    else:
        risk_level = "low"

    # --- Reasoning ---
    if redirect_count == 0:
        reasoning = "The URL resolves directly with no redirects, which is typical for a well-configured site."
    else:
        parts = []
        if redirect_count > 1:
            parts.append(f"redirects {redirect_count} times")
        if cross_domain:
            parts.append(f"lands on a different domain ({final_domain})")
        if http_to_https and not cross_domain:
            parts.append("upgrades the connection to HTTPS")
        if parts:
            reasoning = "The URL " + ", ".join(parts) + "."
        else:
            reasoning = f"The URL redirects once and stays on the same domain ({final_domain})."

    return {
        "status": "ok",
        "input_url": input_url,
        "final_url": final_url,
        "redirect_count": redirect_count,
        "redirect_chain": chain,
        "cross_domain_redirect": cross_domain,
        "http_to_https_upgrade": http_to_https,
        "significant_destination_change": significant_change,
        "risk_level": risk_level,
        "risk_score": score,
        "evidence": evidence,
        "reasoning": reasoning,
    }


def _short(exc):
    """Truncate exception message for display."""
    return str(exc)[:120]
