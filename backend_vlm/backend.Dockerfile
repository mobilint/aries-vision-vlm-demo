FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    ca-certificates && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://dl.mobilint.com/apt/gpg.pub -o /etc/apt/keyrings/mblt.asc && \
    chmod a+r /etc/apt/keyrings/mblt.asc && \
    printf "%s\n" \
      "deb [signed-by=/etc/apt/keyrings/mblt.asc] https://dl.mobilint.com/apt stable multiverse" \
      > /etc/apt/sources.list.d/mobilint.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends mobilint-cli && \
    rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --torch-backend=cpu torch torchaudio torchvision torchcodec && \
    uv pip install --system -r pyproject.toml

CMD ["python", "src/server.py"]
