# ADR 0003: Use a storage-free Azure demo runtime

- Status: Accepted
- Date: 2026-07-13

## Context

The local container originally required a separately mounted UCI-trained `joblib` artifact. A public Azure Container Apps demonstration would therefore need Azure Files or another model distribution mechanism, adding a persistent resource and a secret-bearing artifact path to a deliberately small portfolio deployment.

The reported benchmark must remain tied to the public UCI dataset, while the hosted interface needs only a deterministic model that demonstrates the API and product workflow. Committing or downloading an opaque pickle would weaken the existing deserialization trust boundary.

## Decision

Build a small synthetic TF-IDF and Logistic Regression model inside the container from reviewed source messages. Write a provenance manifest and verify the training-corpus and serialized-artifact SHA-256 checksums before the image can be published. Clearly separate this demonstration model from the UCI benchmark model and metrics.

Publish only versioned, immutable public GHCR images. Deploy their digests to Azure Container Apps Consumption with zero minimum replicas, one maximum replica, the smallest verified resource allocation, the default HTTPS hostname, and no persistent logs. Exclude ACR, Storage/Azure Files, custom DNS, VNet, NAT, private endpoints, and Dedicated workload profiles from the baseline.

Keep `SMS_SPAM_MODEL_PATH` configurable so trusted local artifacts can still be mounted read-only. Add prediction request limits, protected OIDC deployment, an optional budget alert, cost/resource audits, rollback, temporary ingress shutdown, expiry tags, and confirmed resource-group teardown.

## Consequences

- The public demo needs no model storage service or registry credential.
- The model build is reviewable and reproducible; no downloaded pickle enters the image.
- The hosted prediction behavior demonstrates integration but does not inherit the reported UCI benchmark metrics.
- Application and demo-model releases are coupled in the immutable cloud image.
- Scale-to-zero reduces idle compute exposure but introduces cold-start latency.
- The baseline is free-tier-oriented, not guaranteed free; traffic, pricing changes, eligibility, and delayed budget alerts remain financial risks.
- Persisted production telemetry and distributed abuse controls are intentionally absent and would require a separately reviewed architecture.

## Alternatives considered

- **Azure Files mount:** preserves independent model release, but adds a persistent storage account and access configuration for a small demonstration.
- **Azure Container Registry:** integrates naturally with Azure identity, but has a standing registry cost that public GHCR avoids.
- **Commit the UCI-trained pickle:** small and convenient, but exposes a deserialization artifact in source control and blurs model provenance.
- **Train from the full public dataset during every image build:** aligns the hosted model with evaluation, but requires network/data build inputs and makes the release slower and less hermetic.
- **Keep the repository GitHub-only:** has no cloud cost, but does not demonstrate an automated, passwordless deployment path.
