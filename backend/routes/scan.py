from flask import Blueprint, request, jsonify
from services.cert_checker import get_certificate_info
from services.risk_analyzer import analyze_risk
from services.llm_explainer import explain_with_ai
from services.ml_model import load_model, predict_risk
from services.redirect_chain_inspector import inspect_redirect_chain
from services.intermediate_cert_checker import inspect_cert_chain_for_hosts
from urllib.parse import urlparse

scan_bp = Blueprint("scan", __name__)

_model = load_model()


def _extract_domain(url):
    return url.replace("https://", "").replace("http://", "").split("/")[0]


def _nn_score_for(domain):
    prob = predict_risk(domain, _model)
    if prob is None:
        return None
    return round(prob * 100, 1)


@scan_bp.route("/scan", methods=["POST"])
def scan():
    """Cert check + rule-based risk + TextCNN score + redirect chain inspection."""
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
    redirect_result = inspect_redirect_chain(url)

    https_hops = [
        (urlparse(u).hostname, urlparse(u).port or 443)
        for u in redirect_result["redirect_chain"]\
        if urlparse(u).scheme == "https"
    ]
    cert_results = inspect_cert_chain_for_hosts(https_hops)

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
        "intermediate_certs": cert_results,
        "analysis": {
            "findings": risk["findings"],
            "summary": risk["summary"],
        },
        "redirect_check": redirect_result,
    })


@scan_bp.route("/explain", methods=["POST"])
def explain():
    """GPT explanation with cert + NN score + redirect chain context injected."""
    data = request.get_json()
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400

    domain = _extract_domain(url)

    cert_info, error = get_certificate_info(domain)
    if error:
        return jsonify({"error": error}), 400

    risk = analyze_risk(cert_info)
    nn_prob = predict_risk(domain, _model)
    redirect_result = inspect_redirect_chain(url)

    ai = explain_with_ai(domain, cert_info, risk, nn_risk_score=nn_prob, redirect_check=redirect_result)

    return jsonify({
        "ai_explanation": ai["explanation"],
        "recommendations": ai["recommendations"],
        "nn_risk_score": round(nn_prob * 100, 1) if nn_prob is not None else None,
    })
