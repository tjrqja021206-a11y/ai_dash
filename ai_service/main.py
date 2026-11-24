"""FastAPI AI 추론 서버.

실행 (ai_service 폴더에서):
    uvicorn main:app --host 0.0.0.0 --port 8001

- 서버 시작 시 모델을 한 번 로드해 app.state.service에 보관하고 요청마다 재사용합니다.
- 이 서버는 DB에 접근하지 않습니다. 결과 저장은 Django의 Celery Worker가 담당합니다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from routers import predictions
from services.action_inference import ActionInferenceService
from services.model_loader import load_bundle

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = None
    app.state.load_error = None
    try:
        bundle = load_bundle()
        app.state.service = ActionInferenceService(bundle)
        logger.info("모델 준비 완료: %s", bundle.summary()["condition"])
    except Exception as error:  # 모델이 없어도 서버는 떠서 /health로 원인을 보여줌
        app.state.load_error = str(error)
        logger.exception("모델 로드 실패")
    yield


app = FastAPI(
    title="Vision Action AI Service",
    description="YOLO Pose + Random Forest(Motion Capture 17개 동작) 영상 추론 API",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(predictions.router)


@app.get("/health", tags=["health"], summary="서버·모델 준비 상태")
def health(request: Request):
    service = request.app.state.service
    return {
        "status": "ok" if service else "error",
        "model_ready": service is not None,
        "error": request.app.state.load_error,
        "model": service.bundle.summary() if service else None,
    }
