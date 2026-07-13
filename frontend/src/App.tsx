import { FormEvent, useEffect, useId, useState } from "react";
import { ApiError, checkReadiness, predictSms, type TimedPrediction } from "./api";

const EXAMPLES = {
  spam: "Congratulations! You have won a free prize. Claim your reward now.",
  ham: "Hey, I will be there around 7. Should I bring anything for dinner?",
} as const;

type ServiceState = "checking" | "ready" | "unavailable";

function ServiceStatus({ state }: { state: ServiceState }) {
  const copy = {
    checking: "Checking service",
    ready: "Service ready",
    unavailable: "Service unavailable",
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
    <div className="assessment" aria-live="polite" aria-atomic="true">
      <div className={`assessment__status assessment__status--${result.label}`}>
        <span className="assessment__marker" aria-hidden="true">
          {isSpam ? "S" : "H"}
        </span>
        <div>
          <span className="field-label">Classification</span>
          <h2>{isSpam ? "Likely spam" : "Likely legitimate"}</h2>
        </div>
        <span className="classification-tag">{result.label}</span>
      </div>

      <div className="confidence-block">
        <div className="confidence-row">
          <span>Model confidence</span>
          <strong>{confidence}%</strong>
        </div>
        <div
          className={`confidence-track confidence-track--${result.label}`}
          role="progressbar"
          aria-label="Model confidence"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(confidence)}
        >
          <span style={{ width: `${confidence}%` }} />
        </div>
      </div>

      <dl className="result-details">
        <div>
          <dt>Model</dt>
          <dd>TF-IDF + Logistic Regression</dd>
        </div>
        <div>
          <dt>Request time</dt>
          <dd>{result.latencyMs} ms</dd>
        </div>
        <div>
          <dt>Recommended action</dt>
          <dd>{isSpam ? "Review before delivery" : "No automated action"}</dd>
        </div>
      </dl>

      <p className="assessment-note">
        This score supports review decisions; it is not a guarantee that the message is safe or malicious.
      </p>
    </div>
  );
}

function EmptyAssessment() {
  return (
    <div className="empty-assessment">
      <div className="empty-assessment__symbol" aria-hidden="true">—</div>
      <h2>No assessment yet</h2>
      <p>Enter an SMS message and run the classifier to view its label and confidence.</p>
    </div>
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
      <header className="app-header">
        <div className="header-inner">
          <a className="brand" href="#main" aria-label="MessageGuard home">
            <span className="brand-mark" aria-hidden="true">M</span>
            <span>
              <strong>MessageGuard</strong>
              <small>SMS classification service</small>
            </span>
          </a>
          <div className="header-actions">
            <a className="docs-link" href="/docs">API documentation</a>
            <ServiceStatus state={serviceState} />
          </div>
        </div>
      </header>

      <main id="main" className="content">
        <section className="page-heading" aria-labelledby="page-title">
          <div>
            <p className="section-label">Classification workspace</p>
            <h1 id="page-title">SMS risk assessment</h1>
            <p>
              Review one message using the deployed statistical classifier. Results are intended to
              support human decisions, not replace them.
            </p>
          </div>

          <dl className="model-summary" aria-label="Model evaluation summary">
            <div>
              <dt>Spam F1</dt>
              <dd>94.16%</dd>
            </div>
            <div>
              <dt>Evaluation</dt>
              <dd>Duplicate-safe</dd>
            </div>
            <div>
              <dt>Message retention</dt>
              <dd>None</dd>
            </div>
          </dl>
        </section>

        <div className="workspace-grid">
          <section className="panel input-panel" aria-labelledby="classifier-title">
            <div className="panel-header">
              <div>
                <p className="section-label">New request</p>
                <h2 id="classifier-title">Analyze a message</h2>
              </div>
              <span className="endpoint-label">POST /api/v1/predict</span>
            </div>

            <form onSubmit={handleSubmit}>
              <div className="field-heading">
                <label htmlFor={textareaId}>Message text</label>
                <span id={`${textareaId}-hint`} className="character-count">
                  {message.length.toLocaleString()} / 10,000
                </span>
              </div>
              <textarea
                id={textareaId}
                value={message}
                onChange={(event) => {
                  setMessage(event.target.value);
                  if (error) setError("");
                }}
                maxLength={10_000}
                rows={8}
                placeholder="Paste an SMS message for assessment"
                aria-describedby={`${textareaId}-hint ${textareaId}-privacy`}
              />

              <div className="example-row">
                <span>Sample messages:</span>
                <button type="button" className="secondary-button" onClick={() => useExample("spam")}>
                  Suspicious promotion
                </button>
                <button type="button" className="secondary-button" onClick={() => useExample("ham")}>
                  Personal message
                </button>
              </div>

              <p id={`${textareaId}-privacy`} className="privacy-note">
                Message text is processed in memory and excluded from application logs.
              </p>

              {error && <div className="error-message" role="alert">{error}</div>}

              <button className="primary-button" type="submit" disabled={loading}>
                {loading && <span className="spinner" aria-hidden="true" />}
                {loading ? "Running classification…" : "Run classification"}
              </button>
            </form>
          </section>

          <section className="panel result-panel" aria-labelledby="assessment-title">
            <div className="panel-header">
              <div>
                <p className="section-label">Current result</p>
                <h2 id="assessment-title">Assessment</h2>
              </div>
            </div>
            {result ? <ResultCard result={result} /> : <EmptyAssessment />}
          </section>
        </div>

        <section className="evaluation-context" aria-labelledby="evaluation-title">
          <h2 id="evaluation-title">Evaluation context</h2>
          <p>
            The displayed 94.16% spam F1 comes from a grouped split with zero exact message overlap
            between training and test data. The underlying public SMS dataset is historical, so modern
            production use would require monitoring, threshold governance, and periodic retraining.
          </p>
          <a href="/docs">Review the API contract</a>
        </section>
      </main>

      <footer className="app-footer">
        <div>
          <span>Independent portfolio reference implementation</span>
          <span>React · TypeScript · FastAPI · scikit-learn</span>
        </div>
      </footer>
    </div>
  );
}
