import os
import json
from openai import OpenAI

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def explain_with_ai(domain, cert_info, risk, nn_risk_score=None, redirect_check=None):
    """
    Ask OpenAI to explain the scan result in plain English and suggest next steps.

    Parameters
    ----------
    domain         : bare domain string
    cert_info      : dict from cert_checker
    risk           : dict from risk_analyzer
    nn_risk_score  : float [0, 1] from TextCNN, or None
    redirect_check : dict from redirect_chain_inspector, or None

    Returns dict with 'explanation' and 'recommendations'.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "explanation": "AI explanation is disabled. Set the OPENAI_API_KEY environment variable to enable it.",
            "recommendations": ["Add your OpenAI API key to enable AI-powered analysis."],
        }

    # Neural network section
    if nn_risk_score is not None:
        pct = round(nn_risk_score * 100, 1)
        nn_verdict = "HIGH" if pct >= 70 else "MODERATE" if pct >= 40 else "LOW"
        nn_line = (
            f"\n- Neural Network Risk Score: {pct}% ({nn_verdict} phishing risk) — "
            "derived from URL character patterns (subdomains, hyphens, IP usage, length)."
        )
        nn_instruction = (
            "Briefly mention the Neural Network Risk Score and what URL features likely drove it."
        )
    else:
        nn_line = ""
        nn_instruction = ""

    # Redirect chain section
    if redirect_check and redirect_check.get("status") == "ok":
        r = redirect_check
        redirect_line = (
            f"\n- Redirect Chain: {r['redirect_count']} redirect(s). "
            f"Final URL: {r['final_url']}. "
            f"Cross-domain jump: {r['cross_domain_redirect']}. "
            f"HTTP→HTTPS upgrade: {r['http_to_https_upgrade']}. "
            f"Risk level: {r['risk_level']}. "
            f"{r['reasoning']}"
        )
        redirect_instruction = (
            "Include a brief note about the redirect chain findings "
            "(e.g. how many redirects, whether the destination domain changed, "
            "and what that means for the user)."
        )
    elif redirect_check and redirect_check.get("status") == "unavailable":
        redirect_line = f"\n- Redirect Chain: Inspection unavailable ({redirect_check.get('error', 'unknown error')})."
        redirect_instruction = ""
    else:
        redirect_line = ""
        redirect_instruction = ""

    prompt = f"""
A user just scanned the SSL certificate for "{domain}". Here are the results:

- Issuer: {cert_info['issuer']}
- Subject: {cert_info['subject']}
- Valid until: {cert_info['valid_to']} ({cert_info['days_remaining']} days remaining)
- Hostname match: {cert_info['hostname_match']}
- CA verified: {cert_info.get('ca_verified', True)}
- Rule-based risk level: {risk['risk_level']}
- Findings: {", ".join(risk['findings'])}{nn_line}{redirect_line}

Write a 2-3 sentence plain-English explanation of what this means for a non-technical user.
{nn_instruction}
{redirect_instruction}
Then provide 2-3 specific recommended actions.

Respond with JSON only, in this exact format:
{{
  "explanation": "...",
  "recommendations": ["...", "...", "..."]
}}
""".strip()

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "explanation": data.get("explanation", ""),
            "recommendations": data.get("recommendations", []),
        }
    except Exception as e:
        return {
            "explanation": f"AI explanation unavailable: {e}",
            "recommendations": ["Verify your OPENAI_API_KEY and try again."],
        }
