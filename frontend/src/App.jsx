import { useEffect, useState } from "react";
import { fetchHealth, fetchMetrics } from "./api.js";
import TextAnalyzer from "./components/TextAnalyzer.jsx";
import VerdictCard from "./components/VerdictCard.jsx";
import FeatureBreakdown from "./components/FeatureBreakdown.jsx";
import ModelComparison from "./components/ModelComparison.jsx";

export default function App() {
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [connected, setConnected] = useState(false);
  const [apiError, setApiError] = useState(null);

  useEffect(() => {
    fetchMetrics()
      .then(setMetrics)
      .catch(() => setMetrics(null));
    fetchHealth()
      .then(() => setConnected(true))
      .catch((e) => {
        setConnected(false);
        setApiError(e.message);
      });
  }, []);

  return (
    <div className="app">
      <header>
        <h1>AI Text Detector</h1>
        <p className="muted">
          Classifies text as human-written or AI-generated using linguistic
          style features plus TF-IDF — not just keyword matching.
        </p>
      </header>

      {!connected && apiError && (
        <div className="banner-error">
          Can't reach the API ({apiError}). Is the backend running?
        </div>
      )}

      <div className="grid-2">
        <TextAnalyzer onResult={setResult} />
        <VerdictCard result={result} />
      </div>

      <div className="grid-2">
        <FeatureBreakdown features={result?.features} />
        <ModelComparison metrics={metrics} />
      </div>
    </div>
  );
}
