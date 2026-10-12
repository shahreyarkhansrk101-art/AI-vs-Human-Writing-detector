import { useState } from "react";
import { analyzeText, fetchExample } from "../api.js";

const EXAMPLES = [
  { key: "ai_standard", label: "AI example (standard)" },
  { key: "ai_casual", label: "AI example (casual prompt)" },
  { key: "human_casual", label: "Human example (casual)" },
  { key: "human_formal", label: "Human example (formal)" },
];

export default function TextAnalyzer({ onResult }) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleAnalyze() {
    if (text.trim().length < 20) {
      setError("Enter at least 20 characters of text.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await analyzeText(text);
      onResult({ ...result, text });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleExample(subtype) {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchExample(subtype);
      setText(result.text);
      onResult(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <h2>Analyze text</h2>
      <textarea
        className="textarea"
        placeholder="Paste a paragraph of text here (at least 20 characters)..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={8}
      />
      {error && <p className="error-text">{error}</p>}
      <div className="analyzer-actions">
        <button className="btn btn-primary" onClick={handleAnalyze} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
        <span className="muted small">or try an example:</span>
        {EXAMPLES.map((ex) => (
          <button key={ex.key} className="btn" disabled={loading} onClick={() => handleExample(ex.key)}>
            {ex.label}
          </button>
        ))}
      </div>
    </div>
  );
}
