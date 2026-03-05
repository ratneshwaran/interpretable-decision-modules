# Interpretable Decision Modules

A research prototype exploring interpretable decision pipelines that combine rule-based reasoning, fuzzy inference, machine learning predictions, and evidence aggregation.

## Overview

This system is designed to make procurement/risk decisions in an auditable, explainable manner. Rather than relying on a single black-box model, it fuses signals from multiple reasoning layers and produces human-readable explanations alongside every decision.

## Architecture

```
Input Features
      |
      v
+-----+------+     +------------------+     +-------------------+
|  ML Model  | --> |  Fuzzy Inference | --> |  Evidence Fusion  |
| (GBM/LR)   |     |  (risk rules)    |     | (weighted + Dempster)|
+------------+     +------------------+     +-------------------+
                                                     |
                                                     v
                                           +---------+---------+
                                           |  Decision Engine  |
                                           | approve/review/   |
                                           |    reject         |
                                           +---------+---------+
                                                     |
                              +----------------------+---------------------+
                              |                      |                     |
                        Explanation             Audit Trace           Report
                        Bundle                 (JSON log)            (HTML)
```

## Reasoning Layers

### 1. ML Risk Model
A gradient boosting classifier trained on synthetic procurement data. Outputs a risk probability alongside feature importance scores for local explainability.

### 2. Fuzzy Inference System
Encodes domain expertise as linguistic rules:
- `amount` → low / medium / high
- `vendor_risk` → low / medium / high
- `document_confidence` → low / medium / high

Example rule:
> IF amount is HIGH AND document_confidence is LOW THEN risk is HIGH

Fuzzy membership functions allow smooth transitions between categories, avoiding sharp threshold effects.

### 3. Evidence Fusion
Combines signals from the ML model, fuzzy system, and hard rule flags using:
- **Weighted consensus**: linear combination with tunable weights
- **Evidence accumulation**: Dempster-Shafer style belief aggregation

### 4. Interpretability Design
Every decision carries:
- Top ML feature contributions (SHAP-style importance)
- Fuzzy rule activation strengths
- Per-source fusion contribution scores
- Full audit trace stored as timestamped JSON

## Project Structure

```
data/          synthetic dataset generation
models/        ML risk model training and inference
fuzzy/         membership functions and rule engine
fusion/        evidence fusion strategies
explain/       explanation bundle construction
api/           FastAPI decision endpoint
audit/         decision trace logging
reports/       Jinja2 HTML report templates
tests/         unit and integration tests
demo.py        end-to-end pipeline demonstration
```

## Quickstart

```bash
pip install -e ".[dev]"
python demo.py
uvicorn api.app:app --reload
```

## API

```
POST /decision
{
  "amount": 85000,
  "vendor_risk": 0.7,
  "document_confidence": 0.4,
  "anomaly_score": 0.6,
  "historical_risk": 0.5
}
```

Response includes `decision`, `fused_risk`, `explanation`, and `audit_id`.
