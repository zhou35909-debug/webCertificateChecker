def analyze_risk(cert_info):
    """
    Apply rule-based checks to determine risk level, status, and findings.
    Returns a dict with risk_level, status, findings, and summary.
    """
    findings = []
    risk = "Low"
    status = "valid"

    days = cert_info["days_remaining"]

    # --- Expiry checks ---
    if days < 0:
        findings.append(f"Certificate expired {abs(days)} days ago.")
        risk = _escalate(risk, "High")
        status = "invalid"
    elif days < 7:
        findings.append(f"Certificate expires in {days} day(s) — renew immediately.")
        risk = _escalate(risk, "Medium")
        status = "warning"
    elif days < 30:
        findings.append(f"Certificate expires in {days} days — renewal recommended soon.")

    # --- Hostname check ---
    if not cert_info["hostname_match"]:
        findings.append("Hostname does not match the certificate.")
        risk = _escalate(risk, "High")
        status = "invalid"

    # --- CA verification check ---
    # ca_verified=False means the cert was rejected by the system trust store
    # (self-signed, unknown CA, etc.)
    if not cert_info.get("ca_verified", True):
        findings.append("Certificate is not trusted by a public CA (possibly self-signed).")
        risk = _escalate(risk, "Medium")
        if status == "valid":
            status = "warning"

    if not findings:
        findings.append("No issues found. Certificate looks healthy.")

    summaries = {
        "Low":    "Certificate is valid and properly configured.",
        "Medium": "Certificate has issues that should be addressed soon.",
        "High":   "Certificate has critical problems that pose a security risk.",
    }

    return {
        "risk_level": risk,
        "status": status,
        "findings": findings,
        "summary": summaries[risk],
    }


def _escalate(current, new):
    """Return whichever risk level is higher."""
    order = ["Low", "Medium", "High"]
    return new if order.index(new) > order.index(current) else current
