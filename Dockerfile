FROM python:3.12-slim
WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
COPY intel ./intel
COPY agents ./agents
COPY web ./web
COPY data/matches ./data/matches
COPY data/cache ./data/cache
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT}"]
