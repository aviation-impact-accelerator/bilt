FROM ghcr.io/astral-sh/uv:latest AS uv_source
FROM catthehacker/ubuntu:rust-latest

COPY --from=uv_source /uv /uvx /bin/

# Isolate the virtual environment so act's mounting cannot overwrite it
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /github/workspace

COPY . .

RUN uv sync --python 3.14 --all-packages --extra jax
