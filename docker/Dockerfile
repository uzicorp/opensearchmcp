FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y jq curl wget build-essential \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv and create a virtual environment
RUN pip install uv
RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency files and sync
COPY pyproject.toml uv.lock ./
RUN uv sync

COPY . .

ENV PYTHONPATH="/app"

CMD ["python", "main.py"]
