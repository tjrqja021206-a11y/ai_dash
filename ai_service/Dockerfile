# FastAPI + YOLO Pose + Random Forest 추론 이미지 (CPU)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# opencv-python(ultralytics 의존성)이 사용하는 시스템 라이브러리
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && apt-get clean

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import json,urllib.request,sys; d=json.load(urllib.request.urlopen('http://localhost:8001/health')); sys.exit(0 if d['model_ready'] else 1)"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
