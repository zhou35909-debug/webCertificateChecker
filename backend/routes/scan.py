from flask import Blueprint, request, jsonify
from services.cert_checker import get_certificate_info
from services.risk_analyzer import analyze_risk
from services.llm_explainer import explain_with_ai
from services.ml_model import load_model, predict_risk

scan_bp = Blueprint("scan", __name__)

# Load the trained model once at import time.
# Returns None (with a warning) if weights.pth doesn't exist yet.
_model = load_model()


def _extract_domain(url):
    return url.replace("https://", "").replace("http://", "").split("/")[0]


def _nn_score_for(domain):
    """
    Run the PhishingCNN on the bare domain string.
    Returns a percentage float (e.g. 73.4) or None if not yet trained.
    """
    prob = predict_risk(domain, _model)
    if prob is None:
        return None
    return round(prob * 100, 1)


@scan_bp.route("/scan", methods=["POST"])
def scan():
    """Fetch cert + rule-based risk + TextCNN neural network score."""
    data = request.get_json()
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400

    domain = _extract_domain(url)

    cert_info, error = get_certificate_info(domain)
    if error:
        return jsonify({"error": error}), 400

    risk = analyze_risk(cert_info)
    nn_score = _nn_score_for(domain)

    return jsonify({
        "domain": domain,
        "status": risk["status"],
        "risk_level": risk["risk_level"],
        "nn_risk_score": nn_score,
        "certificate": {
            "issuer": cert_info["issuer"],
            "subject": cert_info["subject"],
            "valid_from": cert_info["valid_from"],
            "valid_to": cert_info["valid_to"],
            "days_remaining": cert_info["days_remaining"],
            "hostname_match": cert_info["hostname_match"],
        },
        "analysis": {
            "findings": risk["findings"],
            "summary": risk["summary"],
        },
    })


@scan_bp.route("/explain", methods=["POST"])
def explain():
    """TextCNN score + GPT-4 explanation (score injected into prompt)."""
    data = request.get_json()
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400

    domain = _extract_domain(url)

    cert_info, error = get_certificate_info(domain)
    if error:
        return jsonify({"error": error}), 400

    risk = analyze_risk(cert_info)

    # Raw [0, 1] probability for the LLM prompt
    nn_prob = predict_risk(domain, _model)

    ai = explain_with_ai(domain, cert_info, risk, nn_risk_score=nn_prob)

    return jsonify({
        "ai_explanation": ai["explanation"],
        "recommendations": ai["recommendations"],
        "nn_risk_score": round(nn_prob * 100, 1) if nn_prob is not None else None,
    })
