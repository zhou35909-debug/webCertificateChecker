import { useState } from "react"

function ScanForm({ onScan, loading }) {
  const [url, setUrl] = useState("")

  function handleSubmit(e) {
    e.preventDefault()
    if (url.trim()) onScan(url.trim())
  }

  return (
    <form className="scan-form" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="https://example.com"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        disabled={loading}
      />
      <button type="submit" disabled={loading || !url.trim()}>
        {loading ? "Scanning..." : "Scan"}
      </button>
    </form>
  )
}

export default ScanForm
