export default function VerdictCard({ result }) {
  if (!result) {
    return (
      <div className="panel verdict-panel">
        <h2>Result</h2>
        <p className="muted">Analyze some text or try an example to see a result here.</p>
      </div>
    );
  }

  const { is_ai_generated, confidence, true_label, subtype } = result;
  const pct = Math.round(confidence * 100);
  const verdictClass = is_ai_generated ? "verdict-ai" : "verdict-human";

  return (
    <div className="panel verdict-panel">
      <h2>Result</h2>
      <div className={`verdict-badge ${verdictClass}`}>
        {is_ai_generated ? "Likely AI-generated" : "Likely human-written"}
      </div>
      <div className="confidence-bar-track">
        <div
          className={`confidence-bar-fill ${verdictClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="muted small">{pct}% confidence</p>

      {true_label && (
        <p className="muted small">
          Example ground truth: <strong>{true_label}</strong> ({subtype?.replace(/_/g, " ")})
          {(true_label === "AI") === is_ai_generated ? " — correct" : " — misclassified (a hard case)"}
        </p>
      )}
    </div>
  );
}
