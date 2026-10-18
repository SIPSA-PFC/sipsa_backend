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
│   ├── core/
│   │   └── epiweek.py     ← Brazilian epidemiological weeks
│   └── etl/
│       └── migrate.py     ← loads raw files into the database
├── migrations/            ← Alembic migrations (schema history)
├── alembic.ini
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

# 3. Create your .env file and set SECRET_KEY (required, >= 32 characters)
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # paste into SECRET_KEY=
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

They come from the paper's repository (license CC BY-NC-SA 4.0); see
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for sources, licenses and citations.

The ETL also reads `data/reference/neighborhoods.csv` (committed), which maps each
neighborhood across sources: its row in the ovitrap data, municipal code, health district and
coordinates. See [`data/reference/README.md`](data/reference/README.md).

`python -m src.etl.migrate` can be run again at any time: it updates reference rows in place,
replaces the weekly data and clears cached predictions, all in one transaction.

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
# 1. Create the schema
alembic upgrade head

# 2. Populate the database from raw files
python -m src.etl.migrate

# 3. Create the first account (prompts for a password)
python -m src.cli create-user admin --role admin

# 4. Start the API
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

# 3. Create the schema, run ETL and start API
alembic upgrade head
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

### Train the LSTM models

Forecast routes need trained models. Training runs offline and registers the models in the
database (`model_run`); the API loads them at startup.

```bash
python -m src.ml.train --dev        # 3 repetitions per configuration, ~8 min on 2 CPU cores
python -m src.ml.train              # the paper's 30 repetitions, ~80 min on 2 CPU cores (no GPU needed)
python -m src.ml.evaluate --run data/models/<run_id>   # compare with the paper and naive baselines
```

Models are saved in `data/models/<run_id>/` (not committed). To use an existing run with
another or rebuilt database: `python -m src.ml.train --register data/models/<run_id>`.
How the port compares with the paper: [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

---

## API endpoints

| Method | Endpoint | Auth | DB | Description |
|---|---|---|---|---|
| `POST` | `/auth/token` | No | Yes | Login (form fields `username`, `password`), returns JWT |
| `GET` | `/auth/me` | Yes | Yes | Logged-in username and role |
| `GET` | `/resultados` | No | No | Pre-computed model metrics and predictions |
| `GET` | `/casos/{id}` | Yes | Yes | Weekly dengue case series |
| `GET` | `/ovitrampas/{id}` | Yes | Yes | Weekly OPI + EDI series |
| `GET` | `/predicao` | Yes | Yes | City-wide forecast for the weeks after the latest data (1–4 weeks, by lag) |
| `GET` | `/health` | No | No | Health check |

### Authentication

Send the token from `/auth/token` as `Authorization: Bearer <token>`. Tokens expire after
`ACCESS_TOKEN_EXPIRE_MINUTES` (default 60). Accounts have a role, `admin` or `user`; both
currently have the same access, and `require_role(...)` in `src/api/controllers/auth.py`
restricts a route when that changes.

Manage accounts from the command line (passwords are prompted, never stored in files):

```bash
python -m src.cli create-user NAME [--role admin|user]
python -m src.cli set-password NAME
python -m src.cli set-role NAME admin|user
python -m src.cli deactivate NAME        # existing tokens stop working immediately
python -m src.cli activate NAME
python -m src.cli list-users
```

### Notes per endpoint

**`GET /resultados`** — No parameters, no token, no body. Reads the `.mat` file and returns metrics for all 10 models plus predicted vs actual time series for the two best models. Returns `503` if `Models_1DLSTM_2022_02_04_EvalResults.mat` is not in `data/raw/`.

**`GET /predicao`** — Uses models saved by `python -m src.ml.train` (nothing is trained during a
request). Returns `503` for a configuration with no trained model. The first request for a
configuration loads its saved repetitions (a few seconds); forecasts are stored and reused until
the data is reloaded.

### Query parameters

`/casos/{id}` and `/ovitrampas/{id}` accept optional `year_from` and `year_to` (epidemiological years).

All weekly data uses Brazilian **epidemiological weeks** (Sunday–Saturday, as in SINAN):
each point has `epi_year`, `epi_week` and `week_start` (the Sunday). Week 1 is the first
week with at least 4 days in January, so it can start in late December.

`/predicao` accepts:
- `input_type`: `"dengue"` or `"edi"` (default: `"edi"`)
- `lag`: `1`, `3`, `4`, `5`, or `6` (default: `6`) — any other value returns `422`
- `n_weeks`: weeks ahead, at most the lag window's newest input (lag 1 or 3: 1 week; 4: 2;
  5: 3; 6: 4). Default: that maximum. Forecasts only use data observed up to the latest week,
  so they cannot reach further.

Each forecast point has `predicted_cases` (city total for the week), `lower_bound` /
`upper_bound` (mean ± 1 standard deviation across the model's repetitions — their
disagreement, not a calibrated interval) and `mean_per_neighborhood` (the model's own unit).

### Error responses

| Status | Meaning |
|---|---|
| `401` | Missing or expired token |
| `404` | Neighborhood not found |
| `422` | Invalid query parameter, horizon beyond what the lag allows, or gaps in the data |
| `503` | `.mat` file missing, or no trained model for the requested configuration |

---

## Database schema

Managed with Alembic (`migrations/`). Apply with `alembic upgrade head`; after changing
`src/api/models/db_models.py`, create a migration with
`alembic revision --autogenerate -m "describe change"` and review it before committing.

| Table | Description |
|---|---|
| `data_source` | Where each dataset came from: file or URL, license, citation |
| `district` | Health districts (Norte I, Norte II, Sul, Leste, Oeste) |
| `neighborhood` | The 36 neighborhoods of Natal, RN: official code, district, coordinates and their source |
| `socioeconomic` | 2010 Census snapshot per neighborhood: population, total and per-capita income |
| `epi_week` | Calendar of epidemiological weeks |
| `dengue_case` | Weekly notified cases (whole numbers), one row per neighborhood per week and disease |
| `ovitrap_reading` | Weekly OPI + EDI, one row per neighborhood per week |
| `model_run` | One trained configuration of a training run: training window, metrics, saved model files |
| `forecast` | Stored forecasts per model run, area (city / district / neighborhood), as-of week and target week |

`dengue_case` and `ovitrap_reading` are always queried separately — they represent different measurement systems and are never combined.

> **Existing databases from before Alembic:** delete the SQLite file (or drop the PostgreSQL
> tables), then run `alembic upgrade head` and `python -m src.etl.migrate`.

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
| `data/*.db` | Generated — run `alembic upgrade head` and `python -m src.etl.migrate` to regenerate |
| `data/models/` | Generated by training — run `train.py` to regenerate |
| `outputs/` | Generated by evaluate — run `evaluate.py` to regenerate |
| `.venv/` | Local environment — run `pip install` to regenerate |

---

## License

Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
