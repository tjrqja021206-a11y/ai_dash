"""FastAPI AI 서버 HTTP 호출 (Celery Worker에서 사용)."""

import logging
import time
from pathlib import Path

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """AI 서버 연결 실패·타임아웃·오류 응답."""


def _post_video(url: str, path: Path) -> requests.Response:
    with open(path, "rb") as file:
        return requests.post(
            url,
            files={"file": (Path(path).name, file, "video/mp4")},
            timeout=(10, settings.FASTAPI_TIMEOUT_SECONDS),
        )


def predict_video(path: Path) -> dict:
    """영상을 FastAPI에 보내 추론 결과(JSON)를 받습니다.

    AI 서버가 재시작 중이라 연결이 안 되거나(ConnectionError) 모델을 아직 올리는 중(503)이면
    FASTAPI_RETRIES번까지 잠시 기다렸다가 다시 보냅니다. 시간 초과·잘못된 영상(4xx)은 재시도하지 않습니다.
    """
    url = f"{settings.FASTAPI_URL.rstrip('/')}/predict/video"
    attempts = 1 + max(0, settings.FASTAPI_RETRIES)
    response = None
    for attempt in range(1, attempts + 1):
        try:
            response = _post_video(url, path)
        except requests.ConnectionError as error:
            if attempt == attempts:
                raise AIServiceError(f"AI 서버에 연결할 수 없습니다. ({url})") from error
        except requests.Timeout as error:
            raise AIServiceError(
                f"AI 서버 응답 시간 초과 ({settings.FASTAPI_TIMEOUT_SECONDS}초)"
            ) from error
        else:
            if response.status_code != 503 or attempt == attempts:
                break
        delay = settings.FASTAPI_RETRY_DELAY_SECONDS * attempt
        logger.warning("AI 서버 응답 없음 → %s초 후 재시도 (%s/%s)", delay, attempt, attempts - 1)
        time.sleep(delay)

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise AIServiceError(f"AI 서버 오류 {response.status_code}: {str(detail)[:300]}")
    return response.json()
