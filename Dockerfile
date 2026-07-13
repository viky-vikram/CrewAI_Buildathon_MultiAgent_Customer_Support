# Multi-Agent Customer Support System
#
# Build:  docker build -t support-crew .
# Run:    docker run --rm -p 8501:8501 --env-file .env support-crew
#
# Installs from requirements.txt (bounded version ranges); the CI
# workflow uploads the exact Linux-resolved versions as an artifact
# (requirements-linux.lock) on every run.
FROM python:3.12-slim

# Streamlit needs a writable home for its config; run as non-root.
RUN useradd --create-home appuser
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py pytest.ini pyproject.toml ./
COPY support_crew/ support_crew/
COPY static/ static/
COPY .streamlit/ .streamlit/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Streamlit serves a health endpoint out of the box.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
