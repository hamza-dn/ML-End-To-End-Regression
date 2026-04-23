FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

# xgboost runtime dependency on Debian/Ubuntu slim images
RUN apt-get update \
	&& apt-get install -y --no-install-recommends libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Install API runtime dependencies only (lighter than full project sync).
COPY requirements/docker-api.txt ./requirements/docker-api.txt
RUN uv pip install --system --no-cache -r requirements/docker-api.txt

# Copy only runtime assets required by API inference.
COPY src ./src
COPY data/processed/train_final.csv ./data/processed/train_final.csv
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

