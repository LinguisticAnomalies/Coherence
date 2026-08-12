FROM ghcr.io/astral-sh/uv:0.10.4 AS uvbin
FROM nvidia/cuda:12.8.2-devel-ubuntu24.04 AS builder-gpu

# Install uv.
COPY --from=uvbin /uv /uvx /bin/

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  python3-dev \
  ffmpeg \
  git &&\
  rm -rf /var/lib/apt/lists/*

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app

RUN uv sync --frozen --no-cache

# Download nltk resources
RUN /app/.venv/bin/python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "/app/src/coherencecalculator/scripts/app.py", "--port", "80", "--host", "0.0.0.0"]