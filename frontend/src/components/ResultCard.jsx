const RISK_COLORS = {
  Low: "#22c55e",
  Medium: "#f59e0b",
  High: "#ef4444",
}

function ResultCard({ result }) {
  const { domain, status, risk_level, certificate, analysis, nn_risk_score } = result

  return (
    <div className="result-card">
      <div className="result-header">
        <h2>{domain}</h2>
        <span
          className="risk-badge"
          style={{ background: RISK_COLORS[risk_level] }}
        >
          {risk_level} Risk
        </span>
      </div>

      <div className="cert-grid">
        <Cell label="Status" value={status} />
        <Cell label="Issuer" value={certificate.issuer} />
        <Cell label="Subject" value={certificate.subject} />
        <Cell label="Valid From" value={certificate.valid_from} />
        <Cell label="Valid To" value={certificate.valid_to} />
        <Cell label="Days Remaining" value={certificate.days_remaining} />
        <Cell label="Hostname Match" value={certificate.hostname_match ? "Yes" : "No"} />
      </div>

      {/* Neural Network Risk Score */}
      <NeuralRiskBar score={nn_risk_score} />

      <div className="findings">
        <h3>Findings</h3>
        <ul>
          {analysis.findings.map((f, i) => (
            <li key={i}>{f}</li>
          ))}
        </ul>
      </div>
    </div>
  )
}

function NeuralRiskBar({ score }) {
  if (score === null || score === undefined) {
    return (
      <div className="nn-risk-box nn-risk-untrained">
        <span className="nn-risk-label">Neural Network Score</span>
        <span className="nn-risk-hint">
          Model not trained — run <code>python train_model.py</code> in the backend directory.
        </span>
      </div>
    )
  }

  const color =
    score >= 70 ? "#ef4444" : score >= 40 ? "#f59e0b" : "#22c55e"
  const verdict =
    score >= 70 ? "High Phishing Risk" : score >= 40 ? "Moderate Risk" : "Low Risk"

  return (
    <div className="nn-risk-box">
      <div className="nn-risk-header">
        <span className="nn-risk-label">Neural Network Score</span>
        <span className="nn-risk-verdict" style={{ color }}>
          {score}% — {verdict}
        </span>
      </div>
      <div className="nn-bar-track">
        <div
          className="nn-bar-fill"
          style={{ width: `${score}%`, background: color }}
        />
      </div>
      <p className="nn-risk-caption">
        P(phishing) based on URL structure, SSL state &amp; cert validity
      </p>
    </div>
  )
}

function Cell({ label, value }) {
  return (
    <div className="cert-cell">
      <strong>{label}</strong>
      <span>{value}</span>
    </div>
  )
}

export default ResultCard
