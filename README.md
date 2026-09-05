# ReturnGuard - IEEE-CIS Fraud Detection Baseline

## Overview
ReturnGuard is a fraud detection baseline using corrected fraud detection splits with leakage-free features and proper train/val/test separation. The model predicts whether a transaction is fraudulent based on XGBoost classification.

## Architecture

### Model
- XGBoost classifier trained on 48 engineered features
- Features: TransactionAmt_raw, TransactionAmt_log1p, hour_of_day, day_of_week, is_high_value, txn_deviation_from_customer_avg, card1-6, addr1-2, dist1-2, P_emaildomain, R_emaildomain, V_V12-V_V41
- No target leakage (customer_fraud_rate removed, txn_deviation uses expanding historical mean)

### Data Splits
- **Train**: 354,324 rows (60%)
- **Validation**: 5,999 rows (20%)
- **Test**: 7,321 rows (20%)
- Temporal 60/20/20 split by TransactionDT order

### Baseline Models (validation F1 / test F1 / ROC AUC)
- **Rule-based**: F1≈0.006 / 0.000, AUC≈0.50
- **Logistic Regression**: F1≈0.20 / 0.13, AUC≈0.73 / 0.70
- **XGBoost**: F1≈0.30 / 0.17, AUC≈0.81 / 0.76

### Key Design Decisions
- **No target leakage**: `customer_fraud_rate` removed entirely
- **Fixed feature engineering**: `txn_deviation_from_customer_avg` uses expanding historical mean with `shift(1)` and cold-start fill 0
- **Is_high_value retained**: threshold `TransactionAmt > 500`, documented per `RETENTION_IS_HIGH_VALUE`
- **Class imbalance**: 3.5% fraud rate; handled with `class_weight='balanced'` (logistic) and `scale_pos_weight` (XGBoost)
- **No test tuning**: Validation threshold selected for F1-max; TEST evaluated exactly once

### API
- FastAPI backend at `http://localhost:8000`
- Endpoints: `GET /health`, `POST /predict`
- Prediction response: `risk_score`, `risk_band`, `recommended_action`, `expected_costs`, `top_risk_factors`, `top_protective_factors`, `transaction_amount`

### Decision Engine
- Cost-sensitive approach using expected loss minimization
- Safety constraint: **never autonomously APPROVE** a HIGH-risk transaction; overridden to ESCALATE
- Three actions: APPROVE, HUMAN_REVIEW, ESCALATE

### Explainability
- Perturbation-based feature attribution (SHAP incompatible with NumPy 2.x + XGBoost 3.4.1)
- Top 5 positive and negative contributors with descriptions
- Feature descriptions via `describe_feature()` function

### Setup
1. Clone repository
2. `pip install -r requirements.txt`
3. Start backend: `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000` from the repository root
4. Start frontend: `cd dashboard/frontend` then `npm.cmd run dev` in a second terminal
5. Visit `http://127.0.0.1:5173` to use the dashboard. Keep both terminals running; the Vite development server proxies `/api` to the backend on port 8000.

## Current Status
- ✅ Baseline models trained with proper train/val/test separation
- ✅ Feature engineering pipeline (leakage-fixed)
- ✅ Decision engine with cost matrix
- ✅ Explainability layer (perturbation-based fallback)
- ✅ FastAPI backend (`/health`, `/predict`)
- ✅ Dashboard UI (transaction form + prediction rendering)
- ✅ Git repository initialized
