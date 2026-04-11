import os
import json
from openai import OpenAI

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def explain_with_ai(domain, cert_info, risk, nn_risk_score=None):
    """
    Ask OpenAI to explain the scan result in plain English and suggest next steps.

    Parameters
    ----------
    domain        : bare domain string
    cert_info     : dict from cert_checker
    risk          : dict from risk_analyzer
    nn_risk_score : float [0, 1] — P(phishing) from the local PyTorch model,
                    or None if the model has not been trained yet

    Returns dict with 'explanation' and 'recommendations'.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "explanation": "AI explanation is disabled. Set the OPENAI_API_KEY environment variable to enable it.",
            "recommendations": ["Add your OpenAI API key to enable AI-powered analysis."],
        }

    # ── Build the neural-network section of the prompt ────────────────────
    if nn_risk_score is not None:
        pct = round(nn_risk_score * 100, 1)
        if pct >= 70:
            nn_verdict = "HIGH phishing risk"
        elif pct >= 40:
            nn_verdict = "MODERATE phishing risk"
        else:
            nn_verdict = "LOW phishing risk"

        nn_line = (
            f"\n- Neural Network Risk Score: {pct}% ({nn_verdict}) — "
            "derived from URL structure (length, subdomains, hyphens, IP usage), "
            "SSL certificate state, and cert validity duration."
        )
        nn_instruction = (
            "The local deep-learning model produced a Neural Network Risk Score. "
            "Briefly explain what that score means and which certificate or URL "
            "features most likely drove it (refer to the score value explicitly)."
        )
    else:
        nn_line = ""
        nn_instruction = ""

    prompt = f"""
A user just scanned the SSL certificate for "{domain}". Here are the results:

- Issuer: {cert_info['issuer']}
- Subject: {cert_info['subject']}
- Valid until: {cert_info['valid_to']} ({cert_info['days_remaining']} days remaining)
- Hostname match: {cert_info['hostname_match']}
- CA verified: {cert_info.get('ca_verified', True)}
- Rule-based risk level: {risk['risk_level']}
- Findings: {", ".join(risk['findings'])}{nn_line}

Write a 2-3 sentence plain-English explanation of what this means for a non-technical user.
{nn_instruction}
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
            max_tokens=350,
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
