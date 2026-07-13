# ADR 0001: Serve the existing model through a thin API

- Status: Accepted
- Date: 2026-07-13

## Context

The project already has a tested CLI and a serialized scikit-learn pipeline. A practical portfolio service needs a stable integration contract, readiness behavior, and operational signals without creating a second prediction implementation.

## Decision

Use FastAPI as a thin HTTP adapter around `predict_message`. Configure the trusted model path with `SMS_SPAM_MODEL_PATH` and cache loading by artifact identity so requests do not repeatedly deserialize an unchanged model. Keep training offline and serving stateless. Local benchmark evaluation keeps the UCI-trained artifact outside the image; [ADR 0003](0003-use-a-storage-free-azure-demo-runtime.md) records the later exception for a synthetic, build-generated cloud demo model.

The API provides:

- typed request and response contracts with generated OpenAPI documentation;
- separate liveness and model-readiness semantics;
- JSON request logs containing request ID, route, status, and latency only;
- generic service errors that do not expose local paths;
- a non-root Docker runtime with a trusted model path;
- conservative prediction-body, timeout, and single-instance rate limits.

## Consequences

- The CLI and API use the same prediction and confidence behavior.
- A local benchmark artifact can be replaced independently through the configured mount and path.
- The service stays easy to run locally and explain in an interview.
- Artifact compatibility and provenance must be controlled by the deployer.
- The first request after replacing an artifact pays the model-loading cost.
- This does not provide authentication, distributed rate limiting, a model registry, or automated retraining. The Azure baseline supplies TLS termination and bounded single-replica request controls.

## Alternatives considered

- **Embed the artifact in the image:** simpler startup, but couples code and model releases and would publish a generated binary in the repository/build context.
- **Use a separate model server:** useful for multi-model or high-scale workloads, but unnecessary complexity for this baseline.
- **Add Kubernetes immediately:** does not improve the classifier or its single-service contract; deployment orchestration should follow a demonstrated scaling or availability requirement.
