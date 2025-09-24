FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y jq curl wget build-essential \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

#ARG UV_INDEX_UNITARY_USERNAME
#ARG UV_INDEX_UNITARY_PASSWORD
#
#ENV UV_INDEX_UNITARY_USERNAME=${UV_INDEX_UNITARY_USERNAME}
#ENV UV_INDEX_UNITARY_PASSWORD=${UV_INDEX_UNITARY_PASSWORD}

WORKDIR /app

# Install uv and create a virtual environment
RUN pip install uv
RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency files and sync
COPY pyproject.toml uv.lock ./
RUN uv sync

COPY . .
#COPY ./start.sh /app/start.sh

ENV PYTHONPATH="/app"

#CMD ["./start.sh", "k8s"]
CMD ["python", "main.py"]
