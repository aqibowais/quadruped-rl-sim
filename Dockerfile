FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    ALLOW_RESTART=0

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglfw3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.1 \
    && pip install --no-cache-dir -r requirements-deploy.txt

COPY dashboard ./dashboard
COPY live_server.py randomize_env.py windows_mujoco.py ./

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/setup', timeout=4)"

CMD ["python", "live_server.py", "--no-open"]
