# SIPSA — Dengue Forecasting Backend

Python backend for the SIPSA (Sistema Inteligente de Previsão de Surtos de Arboviroses) project.

Implements a 1D-LSTM model for dengue outbreak forecasting, served via a FastAPI REST API. Originally ported from a MATLAB research pipeline described in:

> *Data-Driven Computational Intelligence Applied to Dengue Outbreak Forecasting: a case study at the scale of the city of Natal, RN-Brazil*

---

## Folder structure

```
dengue-forecasting/
├── data/
│   ├── raw/               ← source files (never committed)
│   ├── models/            ← generated model output (.npz)
│   └── processed/         ← reserved for future use
├── notebooks/
│   └── 02_model_eval.ipynb
├── outputs/               ← saved evaluation figures
├── src/
│   ├── main.py            ← FastAPI entry point
│   ├── core/
│   │   ├── config.py      ← settings loaded from .env
│   │   └── database.py    ← SQLAlchemy async engine (SQLite + PostgreSQL)
│   ├── api/
│   │   ├── controllers/   ← route definitions
│   │   │   ├── auth.py
│   │   │   ├── casos.py
│   │   │   ├── ovitrampas.py
│   │   │   ├── predicao.py
│   │   │   └── resultados.py
│   │   ├── services/      ← business logic
│   │   │   ├── aggregation_service.py
│   │   │   ├── prediction_service.py
│   │   │   ├── resultados_service.py
│   │   │   └── validation_service.py
│   │   ├── repositories/  ← database queries
│   │   │   ├── neighborhood_repository.py
│   │   │   ├── dengue_repository.py
│   │   │   └── ovitrap_repository.py
│   │   └── models/
│   │       ├── db_models.py   ← ORM table definitions
│   │       └── schemas.py     ← Pydantic response schemas
│   ├── ml/
│   │   ├── train.py       ← offline LSTM training
│   │   ├── evaluate.py    ← model evaluation + figures
│   │   ├── inference.py   ← stateful LSTM inference for the API
│   │   └── utils.py       ← shared metrics and data loaders
│   └── etl/
│       └── migrate.py     ← one-time migration from raw files → database
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file
cp .env.example .env
```

---

## Required data files

Place the following in `data/raw/` before running anything.
These files are gitignored and must be copied manually.

| File | Used by |
|---|---|
| `Dados_Modelagem.xlsx` | ETL + training |
| `DADOS GERAIS_01.xlsx` | ETL |
| `Rearranged_Data.mat` | ETL + training |
| `Models_1DLSTM_2022_02_04_EvalResults.mat` | `/resultados` endpoint only |

---

## What to run — by goal

### Just evaluate model coefficients (no database, no API)

Uses the pre-computed MATLAB results. No training required.

```bash
python -m src.ml.evaluate
```

Figures are saved to `outputs/`.

---

### Run the API locally with SQLite

No Docker or PostgreSQL needed. SQLite is the default in `.env.example`.

```bash
# 1. Populate the database from raw files
python -m src.etl.migrate

# 2. Start the API
uvicorn src.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

---

### Run the API with PostgreSQL (production)

```bash
# 1. Start PostgreSQL
docker run --name sipsa-db \
  -e POSTGRES_USER=sipsa \
  -e POSTGRES_PASSWORD=sipsa123 \
  -e POSTGRES_DB=sipsa \
  -p 5432:5432 -d postgres:15

# 2. Update DATABASE_URL in .env
# DATABASE_URL=postgresql+asyncpg://sipsa:sipsa123@localhost:5432/sipsa

# 3. Run ETL and start API
python -m src.etl.migrate
uvicorn src.main:app --reload --port 8000
```

### Switching between SQLite and PostgreSQL

One line in `.env` — nothing else in the code changes:

```bash
# SQLite (local testing — no server needed)
DATABASE_URL=sqlite+aiosqlite:///./data/sipsa_test.db

# PostgreSQL (production)
DATABASE_URL=postgresql+asyncpg://sipsa:sipsa123@localhost:5432/sipsa
```

---

### Retrain the LSTM models from scratch

Only needed to regenerate model weights with new data.
Estimated time: 12–18h on CPU, 1.5–3h on a free Colab T4 GPU.

```bash
python -m src.ml.train
```

---

## API endpoints

| Method | Endpoint | Auth | DB | Description |
|---|---|---|---|---|
| `POST` | `/auth/token` | No | Yes | Login, returns JWT |
| `GET` | `/resultados` | No | No | Pre-computed model metrics and predictions |
| `GET` | `/casos/{id}` | Yes | Yes | Weekly dengue case series |
| `GET` | `/ovitrampas/{id}` | Yes | Yes | Weekly OPI + EDI series |
| `GET` | `/predicao/{id}` | Yes | Yes | LSTM forecast for next 1–12 weeks |
| `GET` | `/health` | No | No | Health check |

### Notes per endpoint

**`GET /resultados`** — No parameters, no token, no body. Reads the `.mat` file and returns metrics for all 10 models plus predicted vs actual time series for the two best models. Returns `503` if `Models_1DLSTM_2022_02_04_EvalResults.mat` is not in `data/raw/`.

**`GET /predicao/{id}`** — Do not call during development. Triggers LSTM training which takes hours. Returns `503` if model weights have not been generated yet. Test `/casos` and `/ovitrampas` first.

### Query parameters

`/casos/{id}` and `/ovitrampas/{id}` accept optional `year_from` and `year_to`.

`/predicao/{id}` accepts:
- `input_type`: `"dengue"` or `"edi"` (default: `"dengue"`)
- `lag`: `1`, `3`, `4`, `5`, or `6` (default: `1`) — any other value returns `422`
- `n_weeks`: `1`–`12` (default: `4`)

### Error responses

| Status | Meaning |
|---|---|
| `401` | Missing or expired token |
| `404` | Neighborhood not found |
| `422` | Invalid query parameter, no geolocation, or insufficient history (< 6 weeks) |
| `503` | `.mat` file missing or model weights not generated yet |

---

## Database schema

Five tables derived directly from the original research data.

| Table | Description |
|---|---|
| `neighborhood` | The 36 neighborhoods of Natal, RN |
| `socioeconomic` | Static income/population snapshot per neighborhood |
| `dengue_case` | Weekly case counts — one row per neighborhood per week |
| `ovitrap_reading` | Weekly OPI + EDI — one row per neighborhood per week |
| `model_prediction` | Cached LSTM output — regenerated automatically on request |

> **Upgrading an existing database:** the `model_prediction` unique constraint now includes
> `input_type` and `lag`. `create_all()` does not alter existing tables, so drop
> `model_prediction` (it is only a cache) or delete the SQLite file and rerun `python -m src.etl.migrate`.

`dengue_case` and `ovitrap_reading` are always queried separately — they represent different measurement systems and are never combined.

---

## MATLAB → Python mapping

| MATLAB | Python |
|---|---|
| `Run1DLSTM_01.m` | `src/ml/train.py` |
| `ModelsPerform_1DLSTM_01.m` | `src/ml/evaluate.py` |
| `ComputeRMSE` | `src/ml/utils.compute_rmse()` |
| `ComputeR` | `src/ml/utils.compute_r()` |
| `TrainLSTM` + `predictAndUpdateState` | `src/ml/inference.py` |

---

## Files that should never be committed

| File / folder | Reason |
|---|---|
| `.env` | Contains secrets and credentials |
| `data/raw/` | Large source files |
| `data/*.db` | Generated by ETL — run `migrate.py` to regenerate |
| `data/models/` | Generated by training — run `train.py` to regenerate |
| `outputs/` | Generated by evaluate — run `evaluate.py` to regenerate |
| `.venv/` | Local environment — run `pip install` to regenerate |

---

## License

Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)