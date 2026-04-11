import ssl
import socket
from datetime import datetime, timezone


def get_certificate_info(domain):
    """
    Fetch SSL certificate info for a domain.
    Returns (cert_dict, error_string). One of the two will be None.
    """
    # Attempt connection with full CA + hostname verification
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain) as tls:
                peer_cert = tls.getpeercert()
        return _parse_peer_cert(peer_cert, domain, hostname_match=True), None

    except ssl.SSLCertVerificationError:
        # Cert failed verification — try to retrieve it anyway for analysis
        return _fetch_unverified(domain)
    except socket.gaierror:
        return None, f"Cannot resolve hostname: {domain}"
    except OSError as e:
        return None, f"Connection failed: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def _fetch_unverified(domain):
    """
    Connect without CA verification to retrieve cert data even if it's
    expired, self-signed, or hostname-mismatched. Requires 'cryptography'.
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        from cryptography.x509.oid import NameOID

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with socket.create_connection((domain, 443), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain) as tls:
                der = tls.getpeercert(binary_form=True)

        cert = x509.load_der_x509_certificate(der, default_backend())

        def get_name(name_obj, attr):
            try:
                return name_obj.get_attributes_for_oid(attr)[0].value
            except Exception:
                return "Unknown"

        issuer = get_name(cert.issuer, NameOID.ORGANIZATION_NAME)
        subject = get_name(cert.subject, NameOID.COMMON_NAME)

        valid_from = _to_naive_utc(getattr(cert, "not_valid_before_utc", cert.not_valid_before))
        valid_to = _to_naive_utc(getattr(cert, "not_valid_after_utc", cert.not_valid_after))
        days_remaining = (valid_to - datetime.utcnow()).days

        try:
            san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            san_list = san_ext.value.get_values_for_type(x509.DNSName)
        except Exception:
            san_list = []

        hostname_match = any(_match_hostname(domain, s) for s in san_list)

        # ca_verified=False means the cert didn't pass CA trust store check
        # (self-signed, unknown CA, etc.)
        return {
            "issuer": issuer,
            "subject": subject,
            "valid_from": valid_from.strftime("%Y-%m-%d"),
            "valid_to": valid_to.strftime("%Y-%m-%d"),
            "days_remaining": days_remaining,
            "hostname_match": hostname_match,
            "ca_verified": False,
        }, None

    except Exception as e:
        return None, f"Cannot parse certificate: {e}"


def _parse_peer_cert(cert, domain, hostname_match):
    """Parse the dict returned by ssl.SSLSocket.getpeercert()."""
    fmt = "%b %d %H:%M:%S %Y %Z"
    valid_from = datetime.strptime(cert["notBefore"], fmt)
    valid_to = datetime.strptime(cert["notAfter"], fmt)
    days_remaining = (valid_to - datetime.utcnow()).days

    issuer = dict(x[0] for x in cert.get("issuer", []))
    subject = dict(x[0] for x in cert.get("subject", []))

    return {
        "issuer": issuer.get("organizationName", "Unknown"),
        "subject": subject.get("commonName", domain),
        "valid_from": valid_from.strftime("%Y-%m-%d"),
        "valid_to": valid_to.strftime("%Y-%m-%d"),
        "days_remaining": days_remaining,
        "hostname_match": hostname_match,
        "ca_verified": True,
    }


def _match_hostname(domain, san):
    """RFC 2818 wildcard matching — * only covers one label, no dots."""
    if san.startswith("*."):
        suffix = san[2:]  # e.g. "badssl.com"
        parts = domain.split(".")
        # wrong.host.badssl.com has 4 parts — parts[1:] = "host.badssl.com" ≠ "badssl.com"
        return len(parts) >= 2 and ".".join(parts[1:]) == suffix
    return domain == san


def _to_naive_utc(dt):
    """Normalize a datetime to timezone-naive UTC (handles both old and new cryptography versions)."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt
