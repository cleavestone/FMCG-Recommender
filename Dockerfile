FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --group serving --group models

COPY src/ src/
COPY data/raw/hh_demographic.csv data/raw/hh_demographic.csv
COPY data/processed/product_dim.parquet data/processed/product_dim.parquet
RUN uv sync --frozen --group serving --group models

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "fmcg_reco.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
