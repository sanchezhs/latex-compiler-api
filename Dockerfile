# Dockerfile
# Build a LaTeX compiler API (FastAPI) with a proper Python venv

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# --- OS deps + TeX Live ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl locales \
        python3 python3-venv python3-pip \
        latexmk biber \
        make git \
        texlive-full && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Locale UTF-8 (optional but nice) ---
RUN sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /srv

# --- Python venv ---
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip inside venv
RUN pip install --upgrade pip

# Copy and install Python deps inside the venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app ./app

# Non-root user
RUN useradd -m runner
USER runner

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

