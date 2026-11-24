"""영상 추론 API.

POST /predict/video       : Celery Worker가 보낸 영상(multipart, 필드명 file) 분석
POST /api/v1/predict/     : 같은 기능 (개발 가이드 권장 경로)
  ?overlay=true           : Skeleton Overlay MP4도 만들어 응답의 overlay.url로 알려줌
GET  /overlay/{id}        : Overlay MP4 내려받기 (받아간 뒤 서버에서 삭제, 1시간 지난 파일도 정리)

응답 status
  ok              : predictions에 사람별 결과가 있음
  no_person       : 영상은 정상이지만 사람이 없음 (HTTP 200, predictions = [])
  no_valid_person : 사람은 있지만 너무 짧거나 가려져 판단 불가 (HTTP 200)
오류(HTTP 4xx/5xx)는 {"detail": "사유"} 형태 → Worker가 FAILURE로 기록
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from functools import partial
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from services.action_inference import VideoTooShortError
from services.overlay import cleanup_old
from services.pose_pipeline import VideoReadError

router = APIRouter(tags=["predictions"])

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
OVERLAY_DIR = Path(os.getenv("OVERLAY_DIR", Path(tempfile.gettempdir()) / "overlays"))
OVERLAY_ID = re.compile(r"^[0-9a-f]{32}$")


async def _predict(request: Request, file: UploadFile, overlay: bool) -> dict:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(503, "모델이 아직 준비되지 않았습니다.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, f"지원하지 않는 파일 형식입니다: '{suffix or '없음'}' (mp4/avi/mov/mkv/webm)"
        )

    # 업로드를 임시 파일로 저장 (OpenCV·YOLO는 파일 경로로 읽음)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = Path(temp.name)
    try:
        size_mb = temp_path.stat().st_size / 1024 / 1024
        if size_mb == 0:
            raise HTTPException(400, "빈 파일입니다.")
        if size_mb > MAX_UPLOAD_MB:
            raise HTTPException(413, f"파일이 너무 큽니다. ({size_mb:.0f}MB > {MAX_UPLOAD_MB}MB)")
        if overlay:
            cleanup_old(OVERLAY_DIR)
        # YOLO 추론은 오래 걸리는 동기 작업 → 이벤트 루프를 막지 않도록 스레드에서 실행
        analyze = partial(service.analyze_video, temp_path, OVERLAY_DIR if overlay else None)
        return await run_in_threadpool(analyze)
    except VideoReadError as error:
        raise HTTPException(400, str(error)) from error
    except VideoTooShortError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        temp_path.unlink(missing_ok=True)


OverlayFlag = Query(False, description="true면 Skeleton Overlay MP4도 생성")


@router.post("/predict/video", summary="영상 한 건의 사람별 동작 추론")
async def predict_video(
    request: Request, file: UploadFile = File(...), overlay: bool = OverlayFlag
):
    return await _predict(request, file, overlay)


@router.post("/api/v1/predict/", summary="영상 추론 (가이드 권장 경로)")
async def predict_video_v1(
    request: Request, file: UploadFile = File(...), overlay: bool = OverlayFlag
):
    return await _predict(request, file, overlay)


@router.get("/overlay/{overlay_id}", summary="Skeleton Overlay MP4 내려받기 (1회)")
def download_overlay(overlay_id: str):
    if not OVERLAY_ID.match(overlay_id):
        raise HTTPException(404, "Overlay 영상이 없습니다.")
    path = OVERLAY_DIR / f"{overlay_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Overlay 영상이 없거나 이미 내려받았습니다.")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"overlay_{overlay_id}.mp4",
        background=BackgroundTask(path.unlink, missing_ok=True),
    )
