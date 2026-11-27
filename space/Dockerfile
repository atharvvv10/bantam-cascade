FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY run /app/run
ENV BANTAM_DEMO_RUN_DIR=/app/run
EXPOSE 7860
CMD ["python", "app.py"]
