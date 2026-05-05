"""
Certificate Chain Inspector — connects to each host in a redirect chain
and returns structured analysis of the TLS certificate at each hop.

Public API
----------
inspect_cert_chain(url) -> dict
inspect_cert_chain_for_hosts(hosts) -> list[dict]

Pair with redirect_chain_inspector.inspect_redirect_chain() to get the
full picture: follow all redirect hops, then call
inspect_cert_chain_for_hosts() with the hostnames from the chain.

Example
-------
    from redirect_chain_inspector import inspect_redirect_chain
    from cert_chain_inspector import inspect_cert_chain_for_hosts
    from urllib.parse import urlparse

    redir = inspect_redirect_chain("http://example.com")
    hosts = [
        (urlparse(u).hostname, urlparse(u).port or 443)
        for u in redir["redirect_chain"]
        if urlparse(u).scheme == "https"
    ]
    cert_results = inspect_cert_chain_for_hosts(hosts)
"""

import datetime
import ipaddress
import socket
import ssl
import warnings
from urllib.parse import urlparse

import urllib.request
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import ExtensionOID
from cryptography.hazmat.primitives import hashes
from cryptography.x509.oid import AuthorityInformationAccessOID
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives.serialization import Encoding

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

_TIMEOUT = 8
_DEFAULT_PORT = 443


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unavailable(host, port, reason):
    return {
        "status": "unavailable",
        "host": host,
        "port": port,
        "error": reason,
        "subject_cn": None,
        "issuer": None,
        "san": [],
        "not_before": None,
        "not_after": None,
        "days_until_expiry": None,
        "expired": None,
        "self_signed": None,
        "chain_depth": None,
        "chain": [],
        "risk_level": "unknown",
        "risk_score": 0.0,
        "evidence": [f"Certificate inspection unavailable: {reason}"],
        "reasoning": f"Could not inspect certificate: {reason}",
    }


def _parse_rdns(rdns_seq):
    """Convert an ssl RDN sequence into a readable dict, e.g. {'CN': 'example.com'}."""
    out = {}
    for rdn in rdns_seq:
        for attr in rdn:
            out[attr[0]] = attr[1]
    return out


def _dt_from_ssl(ssl_date_str):
    """Parse the UTC date string returned by ssl into a datetime object."""
    # Format: 'Apr  1 00:00:00 2025 GMT'
    return datetime.datetime.strptime(ssl_date_str, "%b %d %H:%M:%S %Y %Z").replace(
        tzinfo=datetime.timezone.utc
    )


def _san_list(cert_dict):
    """Extract Subject Alternative Names from the cert dict."""
    sans = []
    for entry in cert_dict.get("subjectAltName", []):
        sans.append(f"{entry[0]}:{entry[1]}")
    return sans


def _fetch_cert_info(host, port):
    """
    Open a TLS connection to host:port and return raw cert dicts for the
    leaf certificate and (where available) the full chain.

    Strategy:
      1. Try with full CA + hostname verification first → ca_verified=True.
      2. On SSLCertVerificationError, retry without verification so we can
         still inspect the cert (expired, self-signed, wrong CA, etc.).

    Returns (leaf_dict, leaf_der, chain_ders, ca_verified).
    leaf_dict is always decoded from DER via cryptography so it is populated
    even when verify_mode=CERT_NONE (where getpeercert() returns {}).
    """
    def _connect(ctx):
        with socket.create_connection((host, port), timeout=_TIMEOUT) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as tls:
                leaf_der = tls.getpeercert(binary_form=True)
                try:
                    chain_certs = tls.get_verified_chain()
                    chain_ders = [c.public_bytes(Encoding.DER) for c in chain_certs]
                except (AttributeError, ImportError):
                    chain_ders = [leaf_der] if leaf_der else []
                return leaf_der, chain_ders

    ca_verified = True
    try:
        verified_ctx = ssl.create_default_context()
        verified_ctx.check_hostname = False   # we check manually
        verified_ctx.verify_mode = ssl.CERT_REQUIRED
        leaf_der, chain_ders = _connect(verified_ctx)
    except ssl.SSLCertVerificationError:
        # Cert exists but failed trust/expiry check — retry to still get data
        ca_verified = False
        unverified_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        unverified_ctx.check_hostname = False
        unverified_ctx.verify_mode = ssl.CERT_NONE
        try:
            leaf_der, chain_ders = _connect(unverified_ctx)
        except Exception as e:
            raise ConnectionError(str(e))
    except socket.timeout:
        raise ConnectionError("Connection timed out")
    except socket.gaierror as e:
        raise ConnectionError(f"DNS resolution failed: {e}")
    except ssl.SSLError as e:
        raise ConnectionError(f"SSL error: {e}")
    except OSError as e:
        raise ConnectionError(str(e))

    if not leaf_der:
        raise ConnectionError("No certificate returned by server")

    # Always decode leaf_dict from DER — getpeercert() returns {} with CERT_NONE
    try:
        cert = x509.load_der_x509_certificate(leaf_der, default_backend())
        leaf_dict = _leaf_dict_from_cryptography(cert)
    except ImportError:
        # Fall back to ssl's own dict (only populated when CERT_REQUIRED)
        leaf_dict = ssl.DER_cert_to_PEM_cert
        leaf_dict = {}
    except Exception:
        leaf_dict = {}

    return leaf_dict, leaf_der, chain_ders, ca_verified


def _leaf_dict_from_cryptography(cert):
    """
    Build a minimal ssl.getpeercert()-compatible dict from a cryptography cert
    so the rest of the inspector can stay unchanged.
    """
    def _rdns(name):
        result = []
        for attr in name:
            short = {
                NameOID.COMMON_NAME: "commonName",
                NameOID.ORGANIZATION_NAME: "organizationName",
                NameOID.COUNTRY_NAME: "countryName",
            }.get(attr.oid, attr.oid.dotted_string)
            result.append(((short, attr.value),))
        return result

    # SANs
    san_list = []
    try:
        ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
        for name in ext.value:
            if isinstance(name, x509.DNSName):
                san_list.append(("DNS", name.value))
            elif isinstance(name, x509.IPAddress):
                san_list.append(("IP Address", str(name.value)))
    except Exception:
        pass

    not_before = getattr(cert, "not_valid_before_utc", cert.not_valid_before)
    not_after  = getattr(cert, "not_valid_after_utc",  cert.not_valid_after)

    if not_before.tzinfo is None:
        not_before = not_before.replace(tzinfo=datetime.timezone.utc)
    if not_after.tzinfo is None:
        not_after = not_after.replace(tzinfo=datetime.timezone.utc)

    fmt = "%b %d %H:%M:%S %Y GMT"
    return {
        "subject":          _rdns(cert.subject),
        "issuer":           _rdns(cert.issuer),
        "notBefore":        not_before.strftime(fmt),
        "notAfter":         not_after.strftime(fmt),
        "subjectAltName":   san_list,
    }


def _aia_chase(leaf_der, max_depth=10):
    """
    Walk up the certificate chain via AIA (Authority Information Access)
    caIssuers URLs, fetching each intermediate until we reach a self-signed
    root or hit max_depth.

    Returns a list of DER bytes: [leaf, intermediate1, intermediate2, ...root?]
    Falls back silently to [leaf_der] if cryptography is missing or any fetch
    fails — callers should always treat the result as best-effort.
    """
    try:

        chain = [leaf_der]
        current_der = leaf_der

        for _ in range(max_depth):
            cert = x509.load_der_x509_certificate(current_der, default_backend())

            if cert.issuer == cert.subject:
                break

            # Extract AIA caIssuers URLs
            try:
                aia = cert.extensions.get_extension_for_oid(
                    ExtensionOID.AUTHORITY_INFORMATION_ACCESS
                )
            except x509.ExtensionNotFound:
                break  # No AIA — can't chase further

            issuer_urls = [
                ad.access_location.value
                for ad in aia.value
                if ad.access_method == AuthorityInformationAccessOID.CA_ISSUERS
            ]

            if not issuer_urls:
                break

            # Fetch the first reachable issuer cert
            fetched = None
            for url in issuer_urls:
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Mozilla/5.0 (compatible; CertAgent/1.0)"}
                    )
                    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                        fetched = resp.read()
                    break
                except Exception:
                    continue

            if not fetched:
                break

            # Some CAs return PEM instead of DER
            if fetched.lstrip().startswith(b"-----BEGIN"):
                import base64
                pem_body = b"".join(fetched.split(b"\n")[1:-2])
                fetched = base64.b64decode(pem_body)

            chain.append(fetched)
            current_der = fetched

        return chain

    except ImportError:
        return [leaf_der]
    except Exception:
        return [leaf_der]


def _parse_chain_depth(chain_ders):
    """
    Try to parse each DER cert in the chain using cryptography (if installed)
    and return summary dicts.  Falls back gracefully if the library is absent.
    """
    summaries = []
    try:
        for der in chain_ders:
            cert = x509.load_der_x509_certificate(der, default_backend())
            subj = cert.subject.get_attributes_for_oid(
                x509.NameOID.COMMON_NAME
            )
            issuer = cert.issuer.get_attributes_for_oid(
                x509.NameOID.COMMON_NAME
            )
            summaries.append({
                "subject_cn": subj[0].value if subj else None,
                "issuer_cn":  issuer[0].value if issuer else None,
                "not_before": cert.not_valid_before_utc.isoformat(),
                "not_after":  cert.not_valid_after_utc.isoformat(),
                "fingerprint_sha256": cert.fingerprint(hashes.SHA256()).hex(),
            })
    except ImportError:
        summaries = [{"note": "Install 'cryptography' for full chain details"}]
    except Exception as e:
        summaries = [{"note": f"Could not parse chain: {e}"}]
    return summaries

def _inspect_single(host, port=_DEFAULT_PORT):
    """
    Inspect the TLS certificate at host:port.

    Returns a result dict with risk scoring.
    """
    if not host:
        return _unavailable(host, port, "No host provided")

    try:
        leaf_dict, leaf_der, chain_ders, ca_verified = _fetch_cert_info(host, port)
    except ConnectionError as e:
        return _unavailable(host, port, str(e))
    except Exception as e:
        return _unavailable(host, port, str(e)[:120])

    if not leaf_dict:
        return _unavailable(host, port, "No certificate returned by server")

    now = datetime.datetime.now(datetime.timezone.utc)

    subject = _parse_rdns(leaf_dict.get("subject", []))
    issuer  = _parse_rdns(leaf_dict.get("issuer",  []))
    subject_cn = subject.get("commonName") or subject.get("CN")
    issuer_cn  = issuer.get("commonName")  or issuer.get("CN")
    issuer_org = issuer.get("organizationName") or issuer.get("O")

    not_before_str = leaf_dict.get("notBefore")
    not_after_str  = leaf_dict.get("notAfter")
    not_before = _dt_from_ssl(not_before_str) if not_before_str else None
    not_after  = _dt_from_ssl(not_after_str)  if not_after_str  else None

    expired = (not_after < now) if not_after else None
    days_until_expiry = (
        (not_after - now).days if (not_after and not expired) else
        (-(now - not_after).days if (not_after and expired) else None)
    )

    san = _san_list(leaf_dict)

    self_signed = (subject_cn == issuer_cn) if (subject_cn and issuer_cn) else None

    if len(chain_ders) <= 1 and leaf_der:
        chain_ders = _aia_chase(leaf_der)

    chain_depth = len(chain_ders)
    chain_summaries = _parse_chain_depth(chain_ders)

    hostname_mismatch = False
    try:
        ssl.match_hostname(leaf_dict, host)
    except ssl.CertificateError:
        hostname_mismatch = True
    except AttributeError:
        # Python 3.12+: match_hostname removed; use a lightweight fallback
        expected = host.lower()
        names = [s.split(":", 1)[1].lower() for s in san if s.startswith("DNS:")]
        if subject_cn:
            names.append(subject_cn.lower())
        hostname_mismatch = not any(_wildcard_match(expected, n) for n in names)

    evidence = []

    if expired:
        evidence.append(
            f"Certificate EXPIRED {abs(days_until_expiry)} day(s) ago "
            f"(expired {not_after.date() if not_after else 'unknown'})."
        )
    elif days_until_expiry is not None and days_until_expiry <= 30:
        evidence.append(
            f"Certificate expires soon: {days_until_expiry} day(s) remaining "
            f"(expires {not_after.date() if not_after else 'unknown'})."
        )
    else:
        if not_after:
            evidence.append(
                f"Certificate valid until {not_after.date()} "
                f"({days_until_expiry} days remaining)."
            )

    if self_signed:
        evidence.append(
            f"Certificate appears self-signed (subject CN == issuer CN: '{issuer_cn}')."
        )
    elif issuer_cn:
        evidence.append(f"Issued by: {issuer_cn}" + (f" ({issuer_org})" if issuer_org else "") + ".")

    if hostname_mismatch:
        evidence.append(
            f"Hostname mismatch: '{host}' does not match the certificate's names."
        )

    if chain_depth > 0:
        evidence.append(f"Certificate chain depth: {chain_depth} certificate(s).")

    if san:
        evidence.append(f"SANs ({len(san)}): {', '.join(san[:8])}" + (" …" if len(san) > 8 else "") + ".")

    if not ca_verified and not self_signed:
        evidence.append("Certificate did not pass CA trust store verification (unknown or untrusted CA).")

    score = 0.0
    if expired:
        score += 0.50
    elif days_until_expiry is not None and days_until_expiry <= 7:
        score += 0.35
    elif days_until_expiry is not None and days_until_expiry <= 30:
        score += 0.20

    if self_signed:
        score += 0.35
    if hostname_mismatch:
        score += 0.30
    if not ca_verified and not self_signed:
        score += 0.25
    if chain_depth == 1 and not self_signed:
        score += 0.10   # only leaf, no intermediates — slightly suspicious

    score = min(round(score, 2), 1.0)

    if score >= 0.60:
        risk_level = "high"
    elif score >= 0.25:
        risk_level = "medium"
    else:
        risk_level = "low"

    # --- Reasoning ---
    issues = []
    if expired:
        issues.append("the certificate has expired")
    if self_signed:
        issues.append("the certificate is self-signed")
    if not ca_verified and not self_signed:
        issues.append("the certificate was not trusted by the CA store")
    if hostname_mismatch:
        issues.append("the hostname does not match the certificate")
    if days_until_expiry is not None and 0 < days_until_expiry <= 30 and not expired:
        issues.append(f"the certificate expires in {days_until_expiry} days")

    if issues:
        reasoning = "TLS risk detected: " + "; ".join(issues) + "."
    else:
        reasoning = (
            f"Certificate for '{host}' appears healthy — valid, not self-signed, "
            "hostname matches, and chain is present."
        )

    return {
        "status": "ok",
        "host": host,
        "port": port,
        "subject_cn": subject_cn,
        "issuer": {"cn": issuer_cn, "org": issuer_org},
        "san": san,
        "not_before": not_before.isoformat() if not_before else None,
        "not_after":  not_after.isoformat()  if not_after  else None,
        "days_until_expiry": days_until_expiry,
        "expired": expired,
        "self_signed": self_signed,
        "ca_verified": ca_verified,
        "hostname_mismatch": hostname_mismatch,
        "chain_depth": chain_depth,
        "chain": chain_summaries,
        "risk_level": risk_level,
        "risk_score": score,
        "evidence": evidence,
        "reasoning": reasoning,
    }


def _wildcard_match(hostname, pattern):
    """Simple wildcard hostname matcher (handles '*.example.com')."""
    if pattern.startswith("*."):
        suffix = pattern[2:]
        parts  = hostname.split(".", 1)
        return len(parts) == 2 and parts[1] == suffix
    return hostname == pattern


def inspect_cert_chain(url):
    """
    Inspect the TLS certificate of the host in *url*.

    Parameters
    ----------
    url : str
        A full URL (e.g. 'https://example.com') or bare hostname.

    Returns
    -------
    dict
        Structured result with risk_level, risk_score, evidence, etc.
    """
    if not url:
        return _unavailable("", _DEFAULT_PORT, "No URL provided")

    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)
    host   = parsed.hostname or ""
    port   = parsed.port or _DEFAULT_PORT

    if parsed.scheme != "https":
        return _unavailable(
            host, port,
            f"Scheme is '{parsed.scheme}', not HTTPS — no TLS certificate to inspect."
        )

    return _inspect_single(host, port)


def inspect_cert_chain_for_hosts(hosts):
    """
    Inspect TLS certificates for multiple hosts, e.g. all HTTPS hops in a
    redirect chain.

    Parameters
    ----------
    hosts : list of (hostname, port) tuples  OR  list of URL strings

    Returns
    -------
    list of result dicts (same schema as inspect_cert_chain)
    """
    results = []
    for entry in hosts:
        if isinstance(entry, str):
            result = inspect_cert_chain(entry)
        else:
            host, port = entry[0], entry[1] if len(entry) > 1 else _DEFAULT_PORT
            result = _inspect_single(host, port)
        results.append(result)
    return results
