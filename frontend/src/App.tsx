import { useState, type FormEvent } from "react";
import type { QueryResponse } from "../types/api";
import { submitQuery } from "../services/api";

export default function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await submitQuery(query);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 760, margin: "0 auto", padding: 24, fontFamily: "sans-serif" }}>
      <h1>FrontierOps AI</h1>
      <p style={{ color: "#666" }}>Enterprise multi-agent RAG platform — ask a policy question.</p>

      <form onSubmit={handleSubmit} style={{ display: "flex", gap: 8, marginBottom: 24 }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. What is the enterprise refund period?"
          style={{ flex: 1, padding: 10, fontSize: 14 }}
        />
        <button type="submit" disabled={loading} style={{ padding: "10px 18px" }}>
          {loading ? "Running agents…" : "Ask"}
        </button>
      </form>

      {error && <div style={{ color: "crimson", marginBottom: 16 }}>{error}</div>}

      {result && (
        <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 16 }}>
          <p>{result.answer}</p>
          <div style={{ fontSize: 13, color: "#666", marginTop: 12 }}>
            <div>Intent: {result.intent}</div>
            <div>Confidence: {(result.confidence * 100).toFixed(1)}%</div>
            <div>
              Status:{" "}
              <strong style={{ color: result.human_review ? "#c0392b" : "#27ae60" }}>
                {result.recommended_action}
              </strong>
            </div>
            <div>Latency: {result.latency_ms.toFixed(1)} ms</div>
          </div>
          {result.evidence.length > 0 && (
            <details style={{ marginTop: 12 }}>
              <summary>Evidence ({result.evidence.length})</summary>
              <ul>
                {result.evidence.map((e) => (
                  <li key={e.chunk_id} style={{ fontSize: 12 }}>
                    {e.chunk_id}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
