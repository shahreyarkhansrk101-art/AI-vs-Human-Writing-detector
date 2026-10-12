const FEATURE_LABELS = {
  num_words: "Word count",
  avg_sentence_len: "Avg sentence length",
  sentence_len_std: "Sentence length variability",
  avg_word_len: "Avg word length",
  type_token_ratio: "Vocabulary richness",
  exclamation_rate: "Exclamations (per 100 words)",
  question_rate: "Questions (per 100 words)",
  contraction_rate: "Contractions (per 100 words)",
  ai_phrase_rate: "AI hedge/transition phrases (per 100 words)",
  first_person_rate: "First-person language (per 100 words)",
  comma_rate: "Commas (per 100 words)",
};

export default function FeatureBreakdown({ features }) {
  if (!features) {
    return (
      <div className="panel">
        <h2>Feature breakdown</h2>
        <p className="muted">Analyze some text to see the linguistic signals behind the score.</p>
      </div>
    );
  }

  return (
    <div className="panel">
      <h2>Feature breakdown</h2>
      <p className="muted small">
        The linguistic signals the model weighs — not just raw word matching.
      </p>
      <div className="feature-list">
        {Object.entries(FEATURE_LABELS).map(([key, label]) => (
          <div className="feature-row" key={key}>
            <span className="feature-label">{label}</span>
            <span className="feature-value">
              {typeof features[key] === "number" ? features[key].toFixed(2) : features[key]}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
