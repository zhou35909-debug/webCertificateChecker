import { useState } from "react"
import ScanForm from "./components/ScanForm"
import ResultCard from "./components/ResultCard"
import AgentExplanation from "./components/AgentExplanation"

const API = import.meta.env.VITE_API_URL ?? "http://localhost:5000"

function App() {
  const [result, setResult] = useState(null)
  const [ai, setAi] = useState(null)
  const [scanning, setScanning] = useState(false)
  const [explaining, setExplaining] = useState(false)
  const [error, setError] = useState(null)

  async function handleScan(url) {
    setScanning(true)
    setExplaining(false)
    setError(null)
    setResult(null)
    setAi(null)

    // Step 1: fetch cert + risk — show result immediately
    try {
      const res = await fetch(`${API}/scan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      })
      const data = await res.json()
      if (!res.ok) {
        setError(data.error || "Scan failed.")
        return
      }
      setResult(data)
    } catch {
      setError("Could not reach the backend. Is the Flask server running?")
      return
    } finally {
      setScanning(false)
    }

    // Step 2: fetch AI explanation separately (non-blocking)
    setExplaining(true)
    try {
      const res = await fetch(`${API}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      })
      const data = await res.json()
      if (res.ok) setAi(data)
    } catch {
      setAi({ ai_explanation: "AI explanation unavailable.", recommendations: [] })
    } finally {
      setExplaining(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>CertAgent</h1>
        <p>AI-powered SSL Certificate Checker</p>
      </header>

      <main>
        <ScanForm onScan={handleScan} loading={scanning} />

        {error && <div className="error-box">{error}</div>}

        {result && <ResultCard result={result} />}

        {(result || explaining) && (
          <AgentExplanation
            explanation={ai?.ai_explanation}
            recommendations={ai?.recommendations}
            loading={explaining}
          />
        )}
      </main>
    </div>
  )
}

export default App
