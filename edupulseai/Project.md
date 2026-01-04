1. Project Purpose (Non-Negotiable)

EduPulse AI is a decision-intelligence system for education providers, designed to detect early student risk, understand sentiment, and forecast enrollment trends in order to enable timely, explainable, and actionable interventions.

The goal is not prediction alone, but insight + action readiness.

2. Core Capabilities (In Scope)

EduPulse AI consists of three primary analytical capabilities:

Student Dropout Early Warning

Predict dropout risk at student level
Produce interpretable risk scores and bands
Highlight key contributing factors for intervention

Student Sentiment Analysis

Analyze textual feedback, surveys, notes, or communications
Detect sentiment polarity and confidence
Aggregate sentiment at cohort / course / term level

Enrollment Projection

Forecast future enrollments and attrition
Identify trends, seasonality, and uncertainty ranges
Support planning and capacity decisions

Each capability must be usable independently, but also composable into a single pipeline.

3. Architectural Principles

The system follows these architectural constraints:

Modular ML design
Each analytical capability is implemented as a standalone module with clear inputs and outputs.

Agent-orchestrated execution
Agents do not contain business logic.
Agents orchestrate, route, and explain results by calling underlying tools/models.

Model-agnostic approach
Multiple model types may coexist (e.g., classical ML, deep learning), but pipelines must remain swappable.

Explainability first
Every prediction must support some form of explanation or rationale.

4. Agent Responsibilities (Strict)

Agents in EduPulse AI are responsible for:

Selecting which analytical tools to invoke

Passing validated inputs to tools

Aggregating outputs into a coherent response

Producing human-readable summaries and recommendations

Agents must not:

Train models

Perform feature engineering

Contain dataset-specific assumptions

Encode institutional policy rules

5. Data Assumptions

Data is provided in structured tabular form (CSV/Parquet)

Text data may be sparse, noisy, or incomplete

No assumption of real-time streaming in MVP

Synthetic or anonymized data is acceptable for development

6. User Interface (MVP Scope)

The MVP includes:

A Streamlit application for:

Uploading datasets

Running analyses

Visualizing results

Exportable outputs (CSV/Parquet) for external BI tools (e.g., Power BI)

Authentication, role-based access, and production hardening are out of scope for MVP.

7. Non-Goals (Explicitly Out of Scope)

The following are intentionally excluded from this project phase:

Real-time inference pipelines

Auto-retraining in production

Institution-specific policy engines

Full MLOps deployment (CI/CD, monitoring, drift detection)

Regulatory or compliance certification

8. Definition of MVP “Done”

The MVP is considered complete when:

Each core capability runs end-to-end on sample data

An agent can orchestrate a multi-step analysis request

Outputs are interpretable and exportable

The Streamlit app demonstrates full workflow without errors

9. Quality Bar

All code must:

Be modular and testable

Avoid hard-coded paths or assumptions

Prefer clarity over cleverness

Follow this contract over ad-hoc feature additions

Any implementation that violates this contract must be revised.

10. Guiding Principle

EduPulse AI is built to answer the question:

“Who needs attention, why, and what should we do next?”

Not merely:
“What does the model predict?”