# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Confirmed:** ReturnGuard is being developed as a public, recruiter-ready software and ML project.
- **Assumption pending confirmation:** The primary in-product user is a fraud or risk analyst reviewing a single transaction before fulfillment.
- **Assumption pending confirmation:** Recruiters and engineers are a secondary evaluation audience who need to understand the system's decisions and technical credibility quickly.

## Product Purpose

ReturnGuard evaluates transaction features with a trained XGBoost model, converts the model score into a cost-sensitive operational recommendation, and explains the main factors behind the result. The rebuild must make that path reliable from a fresh clone through a public deployment while preserving the existing safety rule that high-risk transactions are never approved automatically.

Success means a visitor can submit a realistic sample transaction, receive a validated result, understand the recommended action and its evidence, and verify that the API, tests, and deployment are healthy.

## Positioning

ReturnGuard joins model scoring, business-cost decisioning, and human-readable explanations in one inspectable pipeline. Its distinctive product truth is the separation between prediction and action: the model estimates risk, while a cost-sensitive policy layer recommends APPROVE, HUMAN_REVIEW, or ESCALATE under an explicit human-oversight constraint.

## Operating Context

- A user enters or loads a transaction in the web dashboard.
- The FastAPI service validates the request and runs the saved model artifact.
- The decision engine maps predicted risk and transaction value to an operational action.
- The explanation layer identifies risk-increasing and protective factors.
- The dashboard presents the recommendation, score, costs, and evidence together.

## Capabilities and Constraints

- **Confirmed:** React and Vite frontend, FastAPI backend, XGBoost model, cost-sensitive decision engine, and perturbation-based explanations.
- **Confirmed:** The model was evaluated with a temporal train/validation/test split and target leakage controls.
- **Confirmed:** High-risk transactions must not be approved autonomously.
- **Confirmed:** Existing model and decision behavior should remain explainable and testable as the application is hardened.
- **Assumption pending confirmation:** The first public release is a stateless portfolio-grade demo, with no customer accounts, payment processing, or storage of personal transaction data.
- **Open decision:** Whether a later production-pilot phase should add authentication, persistent case history, analyst assignments, and audit exports.

## Brand Commitments

- Product name: ReturnGuard.
- Voice: technical, direct, calm, and transparent about model limits.
- **Assumption pending confirmation:** The product should feel like a precise fintech operations tool rather than a marketing-heavy AI showcase.

## Evidence on Hand

- Trained model artifacts under `models/`.
- API, decision-engine, feature, and explanation code in the repository.
- A working React transaction-analysis interface under `dashboard/frontend/`.
- Evaluation notes and local tests exist in the working tree, but several are untracked or stale and must not be presented as release evidence until repaired and committed.
- No customer testimonials, live-traffic results, regulatory certification, or production deployment evidence is available; future work must not fabricate them.

## Product Principles

1. Show the decision path, not only a score.
2. Keep human oversight explicit for risky transactions.
3. Prefer reproducible evidence over broad capability claims.
4. Make failure states clear and recoverable.
5. Keep the public demo safe by avoiding personal data and unnecessary persistence.

## Accessibility & Inclusion

**Assumption pending confirmation:** Target WCAG 2.2 AA for the dashboard, including complete keyboard operation, visible focus, sufficient contrast, meaningful status announcements, reduced-motion support, and layouts that remain usable from 320 px mobile width through desktop.
