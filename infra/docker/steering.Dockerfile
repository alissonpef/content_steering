FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY src/ ./src/
COPY config/ ./config/

WORKDIR /app

RUN mkdir -p /app/data/logs/raw /app/data/logs/aggregated /app/data/results

EXPOSE 30500

CMD ["python", "-m", "src.steering.server", "--gateway-mode"]
