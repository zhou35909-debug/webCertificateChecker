import { useState } from "react"

const RISK_COLORS = {
  Low: "#22c55e", low: "#22c55e",
  Medium: "#f59e0b", medium: "#f59e0b",
  High: "#ef4444", high: "#ef4444",
  unknown: "#475569",
}

// ---------------------------------------------------------------------------
// Pipeline container
// ---------------------------------------------------------------------------

function AgentPipeline({ tools, scanResult, aiResult }) {
  return (
    <div className="pipeline">
      <div className="pipeline-header">
        <span className="pipeline-tag">AGENT</span>
        <span className="pipeline-title">Analysis Pipeline</span>
      </div>
      {tools.map(tool => (
        <ToolCard
          key={tool.id}
          tool={tool}
          scanResult={scanResult}
          aiResult={aiResult}
        />
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Individual tool card
// ---------------------------------------------------------------------------

function ToolCard({ tool, scanResult, aiResult }) {
  const [expanded, setExpanded] = useState(false)
  const isDone     = tool.status === "done"
  const canExpand  = isDone

  return (
    <div
      className={`tool-card tool-card--${tool.status}`}
      onClick={canExpand ? () => setExpanded(e => !e) : undefined}
      style={{ cursor: canExpand ? "pointer" : "default" }}
    >
      <div className="tool-card-row">
        <StatusDot status={tool.status} />
        <div className="tool-card-body">
          <span className="tool-name">{tool.name}</span>
          <span className="tool-summary">
            {tool.status === "idle"    && <span className="tool-muted">Pending</span>}
            {tool.status === "running" && <span className="tool-running">Analyzing…</span>}
            {tool.status === "error"   && <span className="tool-error-text">Unavailable</span>}
            {isDone && tool.summary}
          </span>
        </div>
        {canExpand && (
          <span className="tool-chevron">{expanded ? "▲" : "▼"}</span>
        )}
      </div>

      {expanded && isDone && (
        <div className="tool-details">
          <ToolDetails id={tool.id} scanResult={scanResult} aiResult={aiResult} />
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Status dot
// ---------------------------------------------------------------------------

function StatusDot({ status }) {
  return <span className={`status-dot status-dot--${status}`} />
}

// ---------------------------------------------------------------------------
// Expanded detail panels
// ---------------------------------------------------------------------------

function ToolDetails({ id, scanResult, aiResult }) {

  // ── Certificate Checker ──────────────────────────────────────────────────
  if (id === "cert") {
    if (!scanResult) return null
    const { certificate, analysis } = scanResult
    return (
      <div>
        <div className="cert-grid">
          <Cell label="Issuer"         value={certificate.issuer} />
          <Cell label="Subject"        value={certificate.subject} />
          <Cell label="Valid From"     value={certificate.valid_from} />
          <Cell label="Valid To"       value={certificate.valid_to} />
          <Cell label="Days Remaining" value={certificate.days_remaining} />
          <Cell label="Hostname Match" value={certificate.hostname_match ? "Yes" : "No"} />
        </div>
        <div className="findings">
          <h3>Findings</h3>
          <ul>
            {analysis.findings.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      </div>
    )
  }

  // ── URL Phishing Detector ────────────────────────────────────────────────
  if (id === "nn") {
    if (!scanResult) return null
    const score = scanResult.nn_risk_score
    if (score == null) {
      return (
        <div className="nn-risk-box nn-risk-untrained">
          <span className="nn-risk-hint">
            Model not trained — run <code>python train_model.py</code> in the backend directory.
          </span>
        </div>
      )
    }
    const color   = score >= 70 ? "#ef4444" : score >= 40 ? "#f59e0b" : "#22c55e"
    const verdict = score >= 70 ? "High Phishing Risk" : score >= 40 ? "Moderate Risk" : "Low Risk"
    return (
      <div className="nn-risk-box">
        <div className="nn-risk-header">
          <span className="nn-risk-label">Neural Network Score</span>
          <span className="nn-risk-verdict" style={{ color }}>
            {score}% — {verdict}
          </span>
        </div>
        <div className="nn-bar-track">
          <div className="nn-bar-fill" style={{ width: `${score}%`, background: color }} />
        </div>
        <p className="nn-risk-caption">
          P(phishing) based on URL character patterns (subdomains, hyphens, length, IP usage)
        </p>
      </div>
    )
  }

  // ── Redirect Chain Inspector ─────────────────────────────────────────────
  if (id === "redirect") {
    if (!scanResult) return null
    const r = scanResult.redirect_check

    if (!r || r.status === "unavailable") {
      return (
        <div className="redirect-details">
          <p className="redirect-unavailable">
            {r?.error || "Redirect inspection could not be completed."}
          </p>
        </div>
      )
    }

    const riskColor = RISK_COLORS[r.risk_level] || "#94a3b8"
    return (
      <div className="redirect-details">
        <InspectRow label="Input URL"   value={r.input_url} mono />
        {r.final_url !== r.input_url && (
          <InspectRow label="Final URL" value={r.final_url} mono />
        )}
        <InspectRow label="Redirects"   value={r.redirect_count === 0 ? "None" : r.redirect_count} />
        <InspectRow
          label="Cross-Domain"
          value={r.cross_domain_redirect ? "Yes" : "No"}
          valueColor={r.cross_domain_redirect ? "#ef4444" : "#22c55e"}
        />
        <InspectRow
          label="HTTP → HTTPS"
          value={r.http_to_https_upgrade ? "Yes" : "No"}
          valueColor={r.http_to_https_upgrade ? "#22c55e" : undefined}
        />
        <InspectRow
          label="Risk Level"
          value={`${r.risk_level.charAt(0).toUpperCase() + r.risk_level.slice(1)} · ${Math.round(r.risk_score * 100)}%`}
          valueColor={riskColor}
          bold
        />

        <div className="redirect-reasoning">{r.reasoning}</div>

        {r.redirect_chain?.length > 1 && (
          <div className="redirect-chain">
            <span className="redirect-chain-label">Redirect Chain</span>
            <ol>
              {r.redirect_chain.map((hop, i) => (
                <li key={i} className={i === r.redirect_chain.length - 1 ? "redirect-hop--final" : ""}>
                  {hop}
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    )
  }
  if (id === "intermediate") {
    const hostResult = scanResult?.intermediate_certs?.[0]
    const certs = hostResult?.chain ?? []
    if (certs.length === 0) {
      return <p className="details-empty">No intermediate certificate data available.</p>
    }
    return (
      <div className="chain-list">
        {certs.map((cert, i) => {
          const now = new Date()
          const notAfter = cert.not_after ? new Date(cert.not_after) : null
          const expired = notAfter && notAfter < now
          const isLeaf = i === 0
          const isRoot = i === certs.length - 1
          const label = isLeaf ? "Leaf" : isRoot ? "Root" : `Intermediate ${i}`
          return (
            <div key={i} className={`chain-cert${expired ? " chain-cert--expired" : ""}`}>
              <div className="chain-cert-header">
                <span className="chain-cert-index">{i + 1}</span>
                <span className="chain-cert-label">{label}</span>
                {expired && <span className="chain-cert-expired-badge">EXPIRED</span>}
              </div>
              <div className="cert-grid">
                <Cell label="Subject"  value={cert.subject_cn  ?? "—"} />
                <Cell label="Issuer"   value={cert.issuer_cn   ?? "—"} />
                <Cell label="Valid From" value={cert.not_before ? cert.not_before.slice(0, 10) : "—"} />
                <Cell label="Valid To"   value={cert.not_after  ? cert.not_after.slice(0, 10)  : "—"} />
              </div>
              {cert.fingerprint_sha256 && (
                <div className="chain-fingerprint">
                  <span className="inspect-label">SHA-256</span>
                  <span className="inspect-value inspect-value--mono chain-fingerprint-value">
                    {cert.fingerprint_sha256}
                  </span>
                </div>
              )}
              {i < certs.length - 1 && <div className="chain-arrow">↓</div>}
            </div>
          )
        })}
      </div>
    )
  }

  // ── LLM Security Assistant ───────────────────────────────────────────────
  if (id === "llm") {
    if (!aiResult) return <p className="details-empty">No AI analysis available.</p>
    return (
      <div>
        <p className="ai-explanation">{aiResult.ai_explanation}</p>
        {aiResult.recommendations?.length > 0 && (
          <div className="recommendations">
            <h4>Recommended Actions</h4>
            <ul>
              {aiResult.recommendations.map((rec, i) => <li key={i}>{rec}</li>)}
            </ul>
          </div>
        )}
      </div>
    )
  }

  return null
}

// ---------------------------------------------------------------------------
// Helper sub-components
// ---------------------------------------------------------------------------

function Cell({ label, value }) {
  return (
    <div className="cert-cell">
      <strong>{label}</strong>
      <span>{value}</span>
    </div>
  )
}

function InspectRow({ label, value, valueColor, bold, mono }) {
  return (
    <div className="inspect-row">
      <span className="inspect-label">{label}</span>
      <span
        className={`inspect-value${mono ? " inspect-value--mono" : ""}`}
        style={{
          color: valueColor || undefined,
          fontWeight: bold ? 600 : undefined,
        }}
      >
        {value}
      </span>
    </div>
  )
}

export default AgentPipeline
