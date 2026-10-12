const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ? JSON.stringify(body.detail) : `${path} failed: ${res.status}`);
  }
  return res.json();
}

export function fetchHealth() {
  return request("/health");
}

export function fetchMetrics() {
  return request("/metrics");
}

export function analyzeText(text) {
  return request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

export function fetchExample(subtype) {
  return request(`/example/${subtype}`);
}
