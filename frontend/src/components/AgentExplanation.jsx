function AgentExplanation({ explanation, recommendations, loading }) {
  return (
    <div className="ai-card">
      <div className="ai-header">
        <span className="ai-tag">AI</span>
        <h3>Agent Analysis</h3>
      </div>

      {loading ? (
        <p className="ai-loading">Analyzing with AI...</p>
      ) : (
        <>
          <p className="ai-explanation">{explanation}</p>

          {recommendations?.length > 0 && (
            <div className="recommendations">
              <h4>Recommended Actions</h4>
              <ul>
                {recommendations.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default AgentExplanation
