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
# Install dependencies
pip install -e ".[dev]"

# Generate dataset and train model
python data/generate.py
python -c "import sys; sys.path.insert(0,'.'); from models.risk_model import train_and_save; train_and_save()"

# Run end-to-end demo with explanations
python demo.py

# Start the API server
uvicorn api.app:app --reload --port 8000
```

## Running Tests

```bash
pytest tests/ -v
```

## Module Reference

| Module | Purpose |
|--------|---------|
| `data/generate.py` | Synthetic dataset with realistic procurement risk distributions |
| `models/risk_model.py` | GradientBoosting classifier, feature importance extraction |
| `fuzzy/membership.py` | Trapezoidal/shoulder membership functions for 4 features |
| `fuzzy/rules.py` | 10 Mamdani IF-THEN rules, centroid defuzzification |
| `fusion/evidence.py` | Weighted consensus + Dempster-Shafer accumulation |
| `explain/bundle.py` | ExplanationBundle with feature/rule/fusion contributions |
| `audit/logger.py` | JSONL audit log with full decision traces |
| `reports/generator.py` | Jinja2 HTML reports per decision |
| `api/pipeline.py` | Pipeline orchestrator wiring all stages |
| `api/app.py` | FastAPI REST endpoint |

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
