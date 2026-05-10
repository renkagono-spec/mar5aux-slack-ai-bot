FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY slack-app-manifest.example.json .

EXPOSE 8080

CMD ["uvicorn", "slack_ai_bot.app:app", "--host", "0.0.0.0", "--port", "8080"]
