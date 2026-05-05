import { useState } from "react"
import ScanForm from "./components/ScanForm"
import AgentPipeline from "./components/AgentPipeline"

const API = import.meta.env.VITE_API_URL ?? "http://localhost:5000"

const INITIAL_TOOLS = [
  { id: "cert",     name: "Certificate Checker",      status: "idle", summary: null },
  { id: "nn",       name: "URL Phishing Detector",     status: "idle", summary: null },
  { id: "redirect", name: "Redirect Chain Inspector",  status: "idle", summary: null },
  { id: "intermediate", name: "Intermediate Chain Inspector",status: "idle", summary: null },
  { id: "llm",      name: "LLM Security Assistant",    status: "idle", summary: null },
]

function getRedirectSummary(r) {
  if (!r || r.status === "unavailable") return "Could not inspect redirects"
  if (r.redirect_count === 0) return "No redirects"
  const parts = [`${r.redirect_count} redirect${r.redirect_count > 1 ? "s" : ""}`]
  if (r.http_to_https_upgrade) parts.push("upgraded to HTTPS")
  if (r.cross_domain_redirect) parts.push("cross-domain jump")
  return parts.join(" · ")
}

function App() {
  const [tools, setTools]           = useState(INITIAL_TOOLS)
  const [scanResult, setScanResult] = useState(null)
  const [aiResult, setAiResult]     = useState(null)
  const [error, setError]           = useState(null)

  function setToolStatus(id, status, summary = null) {
    setTools(prev => prev.map(t => t.id === id ? { ...t, status, summary } : t))
  }

  async function handleScan(url) {
    setTools(INITIAL_TOOLS.map(t => ({ ...t })))
    setScanResult(null)
    setAiResult(null)
    setError(null)

    // Stagger tools into "running" to convey sequential invocation
    setToolStatus("cert", "running")
    const t1 = setTimeout(() => setToolStatus("nn",       "running"), 450)
    const t2 = setTimeout(() => setToolStatus("redirect", "running"), 900)
    const t3 = setTimeout(() => setToolStatus("intermediate", "running"), 1300)

    let localResult = null
    try {
      const res = await fetch(`${API}/scan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      })
      const data = await res.json()
      clearTimeout(t1)
      clearTimeout(t2)
      clearTimeout(t3);

      if (!res.ok) {
        setError(data.error || "Scan failed.")
        setTools(INITIAL_TOOLS.map(t => ({ ...t })))
        return
      }

      localResult = data
      setScanResult(data)

      const days = data.certificate?.days_remaining
      const daysLabel = days >= 0 ? `${days}d remaining` : "Expired"
      setToolStatus("cert", "done", `${data.risk_level} Risk · ${daysLabel}`)

      const nn = data.nn_risk_score
      setToolStatus("nn", "done",
        nn != null
          ? `${nn}% · ${nn >= 70 ? "High" : nn >= 40 ? "Moderate" : "Low"} Phishing Risk`
          : "Model not trained"
      )

      setToolStatus("redirect", "done", getRedirectSummary(data.redirect_check))
      const certs = data.intermediate_certs?.[0]?.chain ?? []
      const expiredInChain = certs.filter(c => c.not_after && new Date(c.not_after) < new Date())
      const intermediateSummary = certs.length === 0
        ? "No chain data"
        : expiredInChain.length > 0
          ? `${certs.length} certs · ${expiredInChain.length} expired`
          : `${certs.length} cert${certs.length > 1 ? "s" : ""} · Chain OK`
      setToolStatus("intermediate", "done", intermediateSummary)

    } catch {
      clearTimeout(t1)
      clearTimeout(t2)
      setError("Could not reach the backend. Is the Flask server running?")
      setTools(INITIAL_TOOLS.map(t => ({ ...t })))
      return
    }

    // LLM step runs after /scan completes
    setToolStatus("llm", "running")
    try {
      const res = await fetch(`${API}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      })
      const data = await res.json()
      if (res.ok) {
        setAiResult(data)
        setToolStatus("llm", "done", "Analysis complete")
      } else {
        setToolStatus("llm", "error", "AI unavailable")
      }
    } catch {
      setToolStatus("llm", "error", "AI unavailable")
    }
  }

  const isRunning       = tools.some(t => t.status === "running")
  const pipelineVisible = tools.some(t => t.status !== "idle")

  return (
    <div className="app">
      <header>
        <h1>CertAgent</h1>
        <p>AI-powered SSL Certificate &amp; Security Analyzer</p>
      </header>

      <main>
        <ScanForm onScan={handleScan} loading={isRunning} />

        {error && <div className="error-box">{error}</div>}

        {pipelineVisible && (
          <AgentPipeline
            tools={tools}
            scanResult={scanResult}
            aiResult={aiResult}
          />
        )}
      </main>
    </div>
  )
}

export default App
