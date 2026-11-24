"""FastAPI AI 서버 HTTP 호출 (Celery Worker에서 사용)."""

from pathlib import Path

import requests
from django.conf import settings


class AIServiceError(Exception):
    """AI 서버 연결 실패·타임아웃·오류 응답."""


def predict_video(path: Path) -> dict:
    url = f"{settings.FASTAPI_URL.rstrip('/')}/predict/video"
    try:
        with open(path, "rb") as file:
            response = requests.post(
                url,
                files={"file": (Path(path).name, file, "video/mp4")},
                timeout=(10, settings.FASTAPI_TIMEOUT_SECONDS),
            )
    except requests.ConnectionError as error:
        raise AIServiceError(f"AI 서버에 연결할 수 없습니다. ({url})") from error
    except requests.Timeout as error:
        raise AIServiceError(
            f"AI 서버 응답 시간 초과 ({settings.FASTAPI_TIMEOUT_SECONDS}초)"
        ) from error

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise AIServiceError(f"AI 서버 오류 {response.status_code}: {str(detail)[:300]}")
    return response.json()
