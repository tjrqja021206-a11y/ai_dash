# Django 웹 서버와 Celery Worker가 함께 쓰는 이미지
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && apt-get clean

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
# 실제 실행 명령은 compose.yaml의 command에서 서비스별(django / celery)로 지정
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
