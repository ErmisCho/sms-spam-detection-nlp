# ADR 0002: Serve the React demo from the API origin

- Status: Accepted
- Date: 2026-07-13

## Context

The typed prediction API is useful to engineers but does not give recruiters or product reviewers an immediate way to experience the classifier. A small browser interface adds product visibility, but a separately deployed frontend would introduce cross-origin configuration and a second runtime without a demonstrated scaling need.

## Decision

Build a focused React and TypeScript single page with Vite. It uses relative `/api/v1/predict` and `/health/ready` URLs. Vite proxies those paths to FastAPI during development. A Docker build stage compiles the static assets, and FastAPI serves them at `/` from the existing non-root runtime image.

Keep `/docs`, health endpoints, the CLI, and the legacy `/predict` compatibility route available. Local benchmark evaluation mounts the trusted UCI artifact as described in ADR 0001; the hosted storage-free exception is described in ADR 0003.

## Consequences

- Recruiters can understand and try the product without learning Swagger or command-line syntax.
- Production uses one origin, one port, one container, and no CORS allowlist.
- Frontend and backend retain separate source, type, test, and build boundaries.
- The container image build now requires a locked Node dependency stage in addition to the locked Python stage.
- FastAPI static serving is appropriate for this low-traffic portfolio demo; a CDN or dedicated web tier can replace it if traffic or caching requirements justify that infrastructure.
- Authentication, persistence, analytics, and a dashboard remain outside scope because they do not improve the single-message classification workflow.

## Alternatives considered

- **Swagger UI only:** preserves the smallest codebase but is developer documentation rather than a product experience.
- **Separate frontend deployment:** useful for independent scaling and release cadence, but introduces CORS, routing, and another deployment before those needs exist.
- **Next.js:** adds server-side rendering and framework runtime features that this authenticated-free, client-side interaction does not need.
- **Backend-rendered template:** has fewer build tools but demonstrates less frontend engineering and offers a weaker typed component/test workflow.
