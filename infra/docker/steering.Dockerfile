FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y curl && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    rm kubectl

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

COPY src/ ./src/
COPY config/ ./config/

WORKDIR /app

RUN mkdir -p /app/data/logs/raw /app/data/logs/aggregated /app/data/results

EXPOSE 30500

CMD ["python", "-m", "src.steering.server", "--gateway-mode"]
