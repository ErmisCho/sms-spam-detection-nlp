import { FormEvent, useEffect, useId, useState } from "react";
import { ApiError, checkReadiness, predictSms, type TimedPrediction } from "./api";

const EXAMPLES = {
  spam: "Congratulations! You have won a free prize. Claim your reward now.",
  ham: "Hey, I will be there around 7. Should I bring anything for dinner?",
} as const;

type ServiceState = "checking" | "ready" | "unavailable";

function ShieldMark() {
  return (
    <span className="shield" aria-hidden="true">
      <span />
    </span>
  );
}

function ServiceStatus({ state }: { state: ServiceState }) {
  const copy = {
    checking: "Checking model",
    ready: "Model ready",
    unavailable: "Model unavailable",
  }[state];

  return (
    <div className={`service-status service-status--${state}`} role="status">
      <span className="status-dot" aria-hidden="true" />
      {copy}
    </div>
  );
}

function ResultCard({ result }: { result: TimedPrediction }) {
  const isSpam = result.label === "spam";
  const confidence = Math.round(result.confidence * 1000) / 10;

  return (
    <section className={`result result--${result.label}`} aria-live="polite" aria-atomic="true">
      <div className="result__heading">
        <div className="result__icon" aria-hidden="true">
          {isSpam ? "!" : "✓"}
        </div>
        <div>
          <p className="eyebrow">Prediction</p>
          <h2>{isSpam ? "Likely spam" : "Looks legitimate"}</h2>
        </div>
        <span className="result__label">{result.label}</span>
      </div>

      <div className="confidence-row">
        <span>Model confidence</span>
        <strong>{confidence}%</strong>
      </div>
      <div
        className="confidence-track"
        role="progressbar"
        aria-label="Model confidence"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(confidence)}
      >
        <span style={{ width: `${confidence}%` }} />
      </div>

      <div className="result__meta">
        <span>TF-IDF + Logistic Regression</span>
        <span>{result.latencyMs} ms round trip</span>
      </div>
    </section>
  );
}

export default function App() {
  const textareaId = useId();
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<TimedPrediction | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [serviceState, setServiceState] = useState<ServiceState>("checking");

  useEffect(() => {
    const controller = new AbortController();
    void checkReadiness(controller.signal)
      .then((ready) => setServiceState(ready ? "ready" : "unavailable"))
      .catch((caught: unknown) => {
        if (!(caught instanceof DOMException && caught.name === "AbortError")) {
          setServiceState("unavailable");
        }
      });
    return () => controller.abort();
  }, []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = message.trim();
    if (!trimmed) {
      setError("Enter an SMS message before running the classifier.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    try {
      const prediction = await predictSms(trimmed);
      setResult(prediction);
      setServiceState("ready");
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 503) {
        setServiceState("unavailable");
        setError("The model is not ready. Train or mount the model, then try again.");
      } else if (caught instanceof ApiError) {
        setError(caught.message);
      } else {
        setError("The API could not be reached. Check that the service is running.");
      }
    } finally {
      setLoading(false);
    }
  }

  function useExample(kind: keyof typeof EXAMPLES) {
    setMessage(EXAMPLES[kind]);
    setResult(null);
    setError("");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#main" aria-label="MessageGuard home">
          <ShieldMark />
          <span>MessageGuard</span>
        </a>
        <ServiceStatus state={serviceState} />
      </header>

      <main id="main" className="main-grid">
        <section className="intro">
          <p className="kicker">Interpretable NLP · privacy-first demo</p>
          <h1>Know what is safe.<br />Catch what is not.</h1>
          <p className="intro__copy">
            Classify an SMS in milliseconds with a locally trained model. Your message is processed
            in memory and never written to application logs.
          </p>

          <div className="proof-points" aria-label="Model qualities">
            <div>
              <strong>94.16%</strong>
              <span>duplicate-safe spam F1</span>
            </div>
            <div>
              <strong>0</strong>
              <span>exact train/test overlaps</span>
            </div>
            <div>
              <strong>Local</strong>
              <span>default inference path</span>
            </div>
          </div>
        </section>

        <section className="classifier-card" aria-labelledby="classifier-title">
          <div className="card-heading">
            <div>
              <p className="eyebrow">Live classifier</p>
              <h2 id="classifier-title">Check a message</h2>
            </div>
            <span className="api-chip">API connected</span>
          </div>

          <form onSubmit={handleSubmit}>
            <label htmlFor={textareaId}>SMS message</label>
            <div className="textarea-wrap">
              <textarea
                id={textareaId}
                value={message}
                onChange={(event) => {
                  setMessage(event.target.value);
                  if (error) setError("");
                }}
                maxLength={10_000}
                rows={6}
                placeholder="Paste a message here…"
                aria-describedby={`${textareaId}-hint`}
              />
              <span id={`${textareaId}-hint`} className="character-count">
                {message.length.toLocaleString()} / 10,000
              </span>
            </div>

            <div className="example-row">
              <span>Try an example</span>
              <button type="button" className="example-button example-button--spam" onClick={() => useExample("spam")}>
                Suspicious offer
              </button>
              <button type="button" className="example-button" onClick={() => useExample("ham")}>
                Dinner plans
              </button>
            </div>

            {error && <div className="error-message" role="alert">{error}</div>}

            <button className="submit-button" type="submit" disabled={loading}>
              {loading ? <span className="spinner" aria-hidden="true" /> : <span aria-hidden="true">✦</span>}
              {loading ? "Analyzing message…" : "Analyze message"}
            </button>
          </form>

          {result ? (
            <ResultCard result={result} />
          ) : (
            <div className="empty-state" aria-hidden="true">
              <span>01</span>
              <div />
              <span>HAM / SPAM</span>
            </div>
          )}
        </section>
      </main>

      <footer>
        <span>Built with React, TypeScript, FastAPI and scikit-learn</span>
        <a href="/docs">Explore the API ↗</a>
      </footer>
    </div>
  );
}
