FROM vllm/vllm-openai:latest

ENTRYPOINT []

RUN apt update \
    && apt install curl wget vim -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY ["template_alpaca_m.jinja", "."]

RUN pip --no-cache-dir install vllm

COPY ["__init__.py", "decilm.py", "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/"]