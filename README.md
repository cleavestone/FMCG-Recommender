# FMCG Recommender System — End to End

A portfolio project building a hybrid recommender system on real grocery
transaction data: a reproducible modelling pipeline (in notebooks, for
learning) plus a production-shaped FastAPI microservice with OAuth2 auth.

**Dataset:** [Dunnhumby — The Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)
(2,500 households, ~2 years, ~2.6M transaction lines, product hierarchy,
demographics, marketing campaigns, coupons, and promotion-exposure data).

## What it covers

| Area | Technique | FMCG use case |
|---|---|---|
| Baselines | popularity, personal top-N, "buy it again" | reference bar every model must beat |
| Content-based | TF-IDF over product hierarchy | item cold-start (zero purchase history) |
| Collaborative filtering | ALS (`implicit`) | learned taste from purchase behaviour |
| Hybrid | rank-fusion of the above + cold-start fallback | reorder-focused recommendations |
| Market basket analysis | FP-Growth, association rules | basket growth / cross-sell / "complete your basket" |

Two recommendation surfaces are served, deliberately kept separate rather
than blended into one score (see `reports/05_*` and `reports/07_*` for why):
a **reorder** engine (the hybrid) and a **discovery/cross-sell** engine
(basket affinity). Full reasoning and honest results — including where
models *didn't* beat a simpler baseline — are in `reports/`.

## Project layout

```
data/                raw / interim / processed  (gitignored)
notebooks/           numbered, exploratory-to-explanatory (01-07)
reports/             per-notebook findings write-ups
src/fmcg_reco/
  config.py            paths, constants, Settings (env-driven)
  data/                interaction matrix construction
  features/            item content features, association rule mining
  models/              base interface + popularity/content/collaborative/hybrid/affinity
  evaluation/          splitting, metrics, candidate scoping, harness
  artifacts.py         train/serve boundary — save/load fitted models
  serving/             FastAPI app: auth, routers, dependency injection
scripts/             notebook generators + train.py CLI
tests/               unit tests + serving API tests
```

## Setup

```bash
uv sync

uv run python -m ipykernel install --user --name fmcg --display-name "Python (fmcg)"

# Kaggle API token must be at ~/.kaggle/kaggle.json
bash scripts/download_data.sh
```

Run the notebooks in order (01 → 07) to reproduce the modelling pipeline,
or skip straight to training + serving:

```bash
uv run python scripts/train.py          # writes artifacts/latest/{hybrid,affinity}/
cp .env.example .env                    # then set JWT_SECRET / ADMIN_API_KEY
uv run uvicorn fmcg_reco.serving.main:app --reload
```

## API

OAuth2 password grant, JWT bearer tokens. A demo user (`demo` /
`demo-password`) is seeded on first run.

```bash
curl -X POST localhost:8000/auth/token -d "username=demo&password=demo-password"
curl localhost:8000/recommendations/1 -H "Authorization: Bearer <token>"
curl localhost:8000/recommendations/1/complete-basket -H "Authorization: Bearer <token>"
curl localhost:8000/items/<product_id>/similar -H "Authorization: Bearer <token>"
```

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/auth/token` | POST | none | password grant → access + refresh JWT |
| `/auth/refresh` | POST | refresh token | new access token |
| `/auth/me` | GET | access token | whoami |
| `/auth/register` | POST | `X-Admin-Key` header | gated user creation |
| `/health`, `/health/ready` | GET | none | liveness / model-loaded check |
| `/recommendations/{household_id}` | GET | access token | reorder-focused |
| `/recommendations/{household_id}/complete-basket` | GET | access token | cross-sell, with rule explanation |
| `/items/{product_id}/similar` | GET | access token | content-based item similarity |

### Security notes (portfolio-grade, stated explicitly)

- `JWT_SECRET` / `ADMIN_API_KEY` **must** be overridden via `.env` in any
  real deployment — the checked-in defaults are placeholders.
- Refresh tokens are not rotated or revocation-tracked; a stolen refresh
  token is valid until it expires. Production would need a refresh-token
  table with rotation/invalidation.
- No rate limiting on `/auth/token`. Production would add something like
  `slowapi`.
- The service runs plain HTTP; any real deployment needs TLS in front of it.

## Docker

```bash
uv run python scripts/train.py    # artifacts must exist before building
docker build -t fmcg-reco .
docker run -p 8000:8000 --env-file .env -v $(pwd)/artifacts:/app/artifacts fmcg-reco
```

## Testing

```bash
uv run pytest
uv run ruff check .
```

## Roadmap

- **P0** — data audit, cleaning, EDA ✅
- **P1** — evaluation harness + baselines ✅
- **P2** — collaborative filtering + content-based ✅
- **P3** — market basket analysis ✅
- **P4** — learning-to-rank reranker — motivated but not built (see `reports/05_*`: linear rank-fusion collapsed to a single signal, an LTR model could learn conditional trust per household)
- **P5** — uplift modeling + customer analytics — not started
- **P6** — serving API ✅ (Streamlit dashboard not built)
- **P7** — docs site, model cards — not started; this README + `reports/` cover it for now

## Status

Notebooks 01-07 complete, findings in `reports/`. Serving microservice
(FastAPI, OAuth2, two recommendation surfaces, Docker, CI) complete.
